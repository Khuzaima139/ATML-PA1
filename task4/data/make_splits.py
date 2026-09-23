import json
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets

from task4.data.cifar10 import DATA_DIR, SPLIT_FILE

SEED = 6304


def main():
    labels = np.array(datasets.CIFAR10(DATA_DIR, train=True, download=True).targets)

    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text())
        print(f"loaded {SPLIT_FILE}")
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=SEED
        )
        split = {"seed": SEED, "train": sorted(train_idx.tolist()), "val": sorted(val_idx.tolist())}
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(split))
        print(f"created {SPLIT_FILE}")

    assert not set(split["train"]) & set(split["val"])
    assert len(split["train"]) + len(split["val"]) == len(labels)
    for name in ["train", "val"]:
        counts = Counter(labels[split[name]].tolist())
        print(f"{name}: {len(split[name])} images, per class min {min(counts.values())} max {max(counts.values())}")


if __name__ == "__main__":
    main()