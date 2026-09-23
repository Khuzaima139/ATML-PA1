import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from common.seed import get_device, set_seed
from task4.data.cifar10 import cifar10, eval_transform, train_transform
from task4.methods.gcsc import GCSC
from task4.methods.proser import Proser, ProserNet
from task4.methods.vanilla import Vanilla
from task4.models.resnet_cifar import ResNetCIFAR

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints" / "task4"
METHODS = {"vanilla": Vanilla, "gcsc": GCSC}


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load((path.parent / "base.yaml").read_text())
    cfg.update(yaml.safe_load(path.read_text()))
    cfg["run"] = path.stem
    return cfg


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, cfg, shuffle, workers, device):
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg["seed"]),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def accuracy(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i == max_batches:
            break
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return 100 * correct / total


def build(cfg):
    if cfg["method"] != "proser":
        return METHODS[cfg["method"]](), ResNetCIFAR()
    model = ProserNet(cfg["num_dummy"])
    ckpt = torch.load(CKPT_DIR / f"{cfg['init_from']}.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected and sorted(missing) == ["dummy.bias", "dummy.weight"], (missing, unexpected)
    print(f"loaded {cfg['init_from']} checkpoint (epoch {ckpt['epoch']}), dummy head randomly initialised")
    return Proser(cfg), model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    workers = cfg["num_workers"] if device.type == "cuda" else 0

    set_seed(cfg["seed"])
    method, model = build(cfg)
    model = model.to(device)
    print(f"run {cfg['run']} on {device}, workers {workers}, randaugment {method.randaugment}")
    train_loader = make_loader(cifar10("train", train_transform(method.randaugment)), cfg, True, workers, device)
    val_loader = make_loader(cifar10("val", eval_transform()), cfg, False, workers, device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    init_acc = None
    if "init_from" in cfg:
        init_acc = accuracy(model, val_loader, device)
        print(f"initial val acc {init_acc:.2f}", flush=True)

    max_batches = 20 if args.smoke else None
    epochs = 1 if args.smoke else cfg["epochs"]
    ckpt_path = CKPT_DIR / f"{cfg['run']}.pt"
    out_path = ROOT / "task4" / "results" / f"{cfg['run']}_train.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_acc, best_epoch = [], -1.0, 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = opt.param_groups[0]["lr"]
        model.train()
        loss_sum = seen = 0
        sums = defaultdict(float)
        for i, (x, y) in enumerate(train_loader):
            if max_batches and i == max_batches:
                break
            x, y = x.to(device), y.to(device)
            loss = method.loss(model, x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
            for k, v in getattr(method, "logs", {}).items():
                sums[k] += v * len(y)
        sched.step()
        val_acc = accuracy(model, val_loader, device, max_batches)
        row = {"epoch": epoch, "lr": lr, "train_loss": loss_sum / seen,
               **{k: v / seen for k, v in sums.items()},
               "val_acc": val_acc, "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(row, flush=True)
        if args.smoke:
            continue
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, ckpt_path)
            print(f"saved best checkpoint {ckpt_path}", flush=True)
        out_path.write_text(json.dumps(
            {"config": cfg, "init_val_acc": init_acc, "best_epoch": best_epoch, "best_val_acc": best_acc,
             "history": history}, indent=2))


if __name__ == "__main__":
    main()