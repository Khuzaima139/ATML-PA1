import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from common.seed import get_device, set_seed
from shared.pacs import source_dataset, target_dataset
from shared.pacs_protocol import CLASSES, SOURCES
from task2.evaluation.metrics import source_val_metrics
from task2.methods.source_only import SourceOnly
from task2.models.backbone import Backbone, set_train_mode
from task2.models.classifier_head import ClassifierHead

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"source_only": SourceOnly}


def load_config(path):
    cfg = yaml.safe_load((ROOT / "task2" / "configs" / "base.yaml").read_text())
    cfg.update(yaml.safe_load(Path(path).read_text()))
    return cfg


def make_loader(ds, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)


def forever(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    name = Path(args.config).stem

    set_seed(cfg["seed"])
    device = get_device()
    backbone = Backbone().to(device)
    head = ClassifierHead(backbone.out_dim, len(CLASSES)).to(device)
    method = METHODS[cfg["method"]](cfg, backbone.out_dim, len(CLASSES)).to(device)
    model = nn.ModuleDict({"backbone": backbone, "head": head, "method": method})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    src_train = {d: source_dataset(d, "train", train=True) for d in SOURCES}
    src_iters = {d: forever(make_loader(src_train[d], cfg["per_source_batch"], cfg["seed"] + i))
                 for i, d in enumerate(SOURCES)}
    tgt_iter = None
    if method.uses_target:
        tgt_iter = forever(make_loader(target_dataset(train=True), cfg["target_batch"], cfg["seed"] + len(SOURCES)))
    val_loaders = {d: DataLoader(source_dataset(d, "val", train=False), batch_size=64) for d in SOURCES}

    n_src = sum(len(ds) for ds in src_train.values())
    steps_per_epoch = math.ceil(n_src / (cfg["per_source_batch"] * len(SOURCES)))
    total_steps = steps_per_epoch * cfg["max_epochs"]
    print(f"{name}: {steps_per_epoch} updates/epoch, max {cfg['max_epochs']} epochs, device {device}")

    ckpt_path = ROOT / "checkpoints" / "task2" / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, bad = [], -1.0, 0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        start = time.time()
        set_train_mode(model)
        sums = {}
        for step in range(steps_per_epoch):
            xs, ys = zip(*(next(src_iters[d]) for d in SOURCES))
            x, y_s = torch.cat(xs).to(device), torch.cat(ys).to(device)
            n_s = len(x)
            if tgt_iter is not None:
                x = torch.cat([x, next(tgt_iter).to(device)])
            f = backbone(x)
            logits = head(f)
            progress = ((epoch - 1) * steps_per_epoch + step) / total_steps
            loss, logs = method(f[:n_s], logits[:n_s], y_s, f[n_s:], logits[n_s:], progress)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v

        train_logs = {k: v / steps_per_epoch for k, v in sums.items()}
        val = source_val_metrics(backbone, head, val_loaders, device)
        history.append({"epoch": epoch, "train": train_logs, "val": val})

        score = val["mean"]["macro_f1"]
        improved = score > best
        if improved:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "epoch": epoch, "config": cfg}, ckpt_path)
        else:
            bad += 1
        per_domain = " ".join(f"{d} {val[d]['macro_f1']:.2f}" for d in SOURCES)
        losses = " ".join(f"{k} {v:.4f}" for k, v in train_logs.items())
        print(f"epoch {epoch:2d} | {losses} | val F1 {per_domain} | mean {score:.2f}"
              f" | {time.time() - start:.0f}s{' | saved' if improved else ''}")
        if bad >= cfg["patience"]:
            print(f"early stop: {cfg['patience']} epochs without improvement")
            break

    result = {"name": name, "config": cfg, "steps_per_epoch": steps_per_epoch,
              "best_epoch": best_epoch, "stopped_epoch": epoch,
              "best_val": history[best_epoch - 1]["val"], "history": history}
    out = ROOT / "task2" / "results" / f"{name}_train.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"best epoch {best_epoch}, mean source-val macro-F1 {best:.2f}; wrote {out.name}")


if __name__ == "__main__":
    main()