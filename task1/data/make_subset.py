import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load(open(ROOT / "task1/configs/task1.yaml"))
SEED = cfg["seed"]


def main():
    train_set = STL10(ROOT / cfg["data_root"], split="train", download=True)
    test_set = STL10(ROOT / cfg["data_root"], split="test", download=True)
    classes = train_set.classes

    # 80/20 stratified split of the official train set
    all_idx = np.arange(len(train_set.labels))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=cfg["val_fraction"],
        stratify=train_set.labels,
        random_state=SEED,
    )

    # 50 test images per class
    rng = np.random.default_rng(SEED)
    eval_idx = []
    for c in range(len(classes)):
        class_idx = np.where(test_set.labels == c)[0]
        n = min(cfg["eval_per_class"], len(class_idx))
        chosen = rng.choice(class_idx, size=n, replace=False)
        eval_idx.extend(chosen.tolist())

    split = {
        "seed": SEED,
        "classes": classes,
        "train_idx": sorted(train_idx.tolist()),
        "val_idx": sorted(val_idx.tolist()),
        "eval_idx": sorted(eval_idx),
    }
    out_path = ROOT / cfg["split_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f)

    print("train per class:", np.bincount(train_set.labels[train_idx]).tolist())
    print("val per class:  ", np.bincount(train_set.labels[val_idx]).tolist())
    print("eval per class: ", np.bincount(test_set.labels[eval_idx]).tolist())
    print("saved to", out_path)


if __name__ == "__main__":
    main()