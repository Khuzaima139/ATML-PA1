import json

import torch
from PIL import Image
from torch.utils.data import Dataset

from common.seed import get_device
from task1.data.stl10 import BASE_TRANSFORM, ROOT
from task1.models.backbones import load_backbones
from task1.scripts.extract_features import extract

CACHE = ROOT / "task1/cache"
FINAL = ROOT / "task1/cache/cue_conflicts_final"


class CueConflictDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        m = self.manifest[idx]
        img = Image.open(FINAL / f"{m['id']}.png").convert("RGB")
        return BASE_TRANSFORM(img), 0  # label unused, kept for extract()'s (x, y) signature


def main():
    device = get_device()
    with open(ROOT / "task1/results/cue_conflicts_final.json") as f:
        final = json.load(f)
    manifest = final["images"]

    dataset = CueConflictDataset(manifest)
    backbones, _, _ = load_backbones(device)

    for name, backbone in backbones.items():
        path = CACHE / f"{name}_cue_conflict.pt"
        if path.exists():
            print(f"loaded {path.name}")
            continue
        feats, _ = extract(backbone, dataset, device)
        torch.save({"feats": feats, "manifest": manifest}, path)
        print(f"extracted {path.name}: {tuple(feats.shape)}")


if __name__ == "__main__":
    main()