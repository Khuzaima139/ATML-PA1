import copy
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.seed import set_seed
from task1.data.stl10 import ROOT, cfg

CACHE = ROOT / "task1/cache"
CKPT = ROOT / "checkpoints/task1"
RESULTS = ROOT / "task1/results"
BACKBONES = ["resnet50", "vit_b16", "clip_b32"]


def load(name, split):
    d = torch.load(CACHE / f"{name}_{split}_clean.pt")
    return d["feats"], d["labels"]


@torch.no_grad()
def accuracy(head, x, y):
    return 100 * (head(x).argmax(dim=1) == y).float().mean().item()


def train_head(name, hp):
    set_seed()
    x_train, y_train = load(name, "train")
    x_val, y_val = load(name, "val")

    head = nn.Linear(x_train.shape[1], 10)
    optimizer = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=hp["batch_size"], shuffle=True)

    best_acc, best_epoch, best_state = -1, 0, None
    epochs_without_improvement = 0
    log = []

    for epoch in range(1, hp["max_epochs"] + 1):
        head.train()
        total_loss = 0
        for xb, yb in loader:
            loss = loss_fn(head(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        head.eval()
        train_loss = total_loss / len(x_train)
        val_acc = accuracy(head, x_val, y_val)
        log.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= hp["patience"]:
            break

    head.load_state_dict(best_state)
    print(f"{name}: best val acc {best_acc:.2f}% at epoch {best_epoch}, stopped at epoch {epoch}")
    return head, {"best_val_acc": best_acc, "best_epoch": best_epoch, "stopped_epoch": epoch, "log": log}


def main():
    hp = cfg["head"]
    CKPT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {"hyperparameters": hp, "seed": cfg["seed"], "heads": {}}

    for name in BACKBONES:
        head, info = train_head(name, hp)
        torch.save(head.state_dict(), CKPT / f"{name}_head.pt")
        summary["heads"][name] = info

    with open(RESULTS / "head_training.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()