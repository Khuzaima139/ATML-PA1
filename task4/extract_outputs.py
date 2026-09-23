import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.seed import get_device
from task4.data.cifar10 import cifar10, eval_transform
from task4.data.cifar100_unknowns import unknowns
from task4.methods.proser import ProserNet
from task4.models.resnet_cifar import ResNetCIFAR

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints" / "task4"
CACHE_DIR = ROOT / "task4" / "cache"
SPLITS = ["train", "val", "test", "near", "far"]


def load_model(run, device):
    ckpt = torch.load(CKPT_DIR / f"{run}.pt", map_location="cpu")
    cfg = ckpt["config"]
    model = ProserNet(cfg["num_dummy"]) if cfg["method"] == "proser" else ResNetCIFAR()
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["epoch"]


def dataset(split):
    if split in ("near", "far"):
        return unknowns(split, eval_transform())
    return cifar10(split, eval_transform())


@torch.no_grad()
def extract(model, ds, device):
    out = {"feat": [], "logits": [], "labels": [], "dummy": []}
    for x, y in DataLoader(ds, batch_size=500, shuffle=False):
        f = model.features(x.to(device))
        out["feat"].append(f.cpu())
        out["logits"].append(model.net.fc(f).cpu())
        out["labels"].append(y)
        if isinstance(model, ProserNet):
            out["dummy"].append(model.dummy(f).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def load_outputs(run):
    return dict(np.load(CACHE_DIR / f"{run}.npz"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    path = CACHE_DIR / f"{args.run}.npz"
    if path.exists():
        cache = load_outputs(args.run)
        print(f"loaded {path}")
    else:
        device = get_device()
        model, epoch = load_model(args.run, device)
        print(f"extracting {args.run} (checkpoint epoch {epoch}) on {device}")
        cache = {}
        for split in SPLITS:
            for k, v in extract(model, dataset(split), device).items():
                cache[f"{split}_{k}"] = v
            print(f"{split}: {len(cache[f'{split}_labels'])} images")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cache)
        print(f"created {path}")

    for split in ["val", "test"]:
        acc = 100 * (cache[f"{split}_logits"].argmax(1) == cache[f"{split}_labels"]).mean()
        print(f"CIFAR-10 {split} accuracy {acc:.2f}")


if __name__ == "__main__":
    main()