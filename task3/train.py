import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from common.seed import get_device, set_seed
from shared.pacs import source_dataset
from shared.pacs_protocol import CLASSES, SOURCES
from task2.train import forever, load_config, make_loader
from task3.methods.dan_dg import DANDG
from task3.methods.erm import ERM
from task3.methods.sam import SAM
from task3.models.backbone import Backbone, set_train_mode
from task3.models.classifier_head import ClassifierHead
from task3.selection.source_validation import evaluate_sources, selection_score, val_loaders

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"erm": ERM, "dan_dg": DANDG}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    name = Path(args.config).stem
    assert "checkpoint" not in cfg, "ERM is reused from Task 2, not retrained"

    set_seed(cfg["seed"])
    device = get_device()
    backbone = Backbone().to(device)
    head = ClassifierHead(backbone.out_dim, len(CLASSES)).to(device)
    method = METHODS[cfg["method"]](cfg, backbone.out_dim, len(CLASSES)).to(device)
    assert not method.uses_target
    model = nn.ModuleDict({"backbone": backbone, "head": head, "method": method})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sam = SAM(model.parameters(), opt, cfg["rho"]) if "rho" in cfg else None

    src_train = {d: source_dataset(d, "train", train=True) for d in SOURCES}
    src_iters = {d: forever(make_loader(src_train[d], cfg["per_source_batch"], cfg["seed"] + i))
                 for i, d in enumerate(SOURCES)}
    loaders = val_loaders()

    n_src = sum(len(ds) for ds in src_train.values())
    steps_per_epoch = math.ceil(n_src / (cfg["per_source_batch"] * len(SOURCES)))
    total_steps = steps_per_epoch * cfg["max_epochs"]
    print(f"{name}: {steps_per_epoch} updates/epoch, max {cfg['max_epochs']} epochs, device {device}"
          f"{', SAM rho ' + str(cfg['rho']) if sam else ''}")

    ckpt_path = ROOT / "checkpoints" / "task3" / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, bad = [], -1.0, 0, 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        start = time.time()
        set_train_mode(model)
        sums = {}
        for step in range(steps_per_epoch):
            xs, ys = zip(*(next(src_iters[d]) for d in SOURCES))
            x, y = torch.cat(xs).to(device), torch.cat(ys).to(device)
            progress = ((epoch - 1) * steps_per_epoch + step) / total_steps

            def compute():
                f = backbone(x)
                logits = head(f)
                return method(f, logits, y, f[:0], logits[:0], progress)

            if sam is None:
                loss, logs = compute()
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                loss, logs = sam.step(compute)
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v

        train_logs = {k: v / steps_per_epoch for k, v in sums.items()}
        val = evaluate_sources(backbone, head, loaders, device)
        history.append({"epoch": epoch, "train": train_logs, "val": val})

        score = selection_score(val)
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
    out = ROOT / "task3" / "results" / f"{name}_train.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"best epoch {best_epoch}, mean source-val macro-F1 {best:.2f}; wrote {out.name}")


if __name__ == "__main__":
    main()