import json
from pathlib import Path

import yaml
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.datasets import STL10

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load(open(ROOT / "task1/configs/task1.yaml"))

BASE_TRANSFORM = T.Compose([
    T.Resize((cfg["image_size"], cfg["image_size"]), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
])


def get_sets():
    with open(ROOT / cfg["split_file"]) as f:
        split = json.load(f)
    train_full = STL10(ROOT / cfg["data_root"], split="train", transform=BASE_TRANSFORM)
    test_full = STL10(ROOT / cfg["data_root"], split="test", transform=BASE_TRANSFORM)
    sets = {
        "train": Subset(train_full, split["train_idx"]),
        "val": Subset(train_full, split["val_idx"]),
        "eval": Subset(test_full, split["eval_idx"]),
    }
    return sets, split["classes"]

def load_style_pool():
    with open(ROOT / "task1/data/style_candidates.json") as f:
        candidates = json.load(f)
    with open(ROOT / "task1/data/style_pool.json") as f:
        picks = json.load(f)
    return {name: [candidates[name][k] for k in picks[name]] for name in picks}