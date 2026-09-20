import json

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import STL10

from task1.data.stl10 import ROOT, cfg

OUT_DIR = ROOT / "task1/cache/style_candidates"


def main():
    with open(ROOT / cfg["split_file"]) as f:
        split = json.load(f)
    classes = split["classes"]
    train_idx = np.array(split["train_idx"])

    dataset = STL10(ROOT / cfg["data_root"], split="train")
    labels = dataset.labels[train_idx]
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["cue_conflict"]["style_candidates"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = {}
    for c, name in enumerate(classes):
        chosen = rng.choice(train_idx[labels == c], size=n, replace=False)
        candidates[name] = [int(i) for i in chosen]
        fig, axes = plt.subplots(5, 6, figsize=(12, 10))
        for k, (ax, idx) in enumerate(zip(axes.flat, candidates[name])):
            ax.imshow(dataset[idx][0])
            ax.set_title(str(k))
            ax.axis("off")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}.png", dpi=100)
        plt.close(fig)
        print(f"saved {name}.png")

    with open(ROOT / "task1/data/style_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)


if __name__ == "__main__":
    main()