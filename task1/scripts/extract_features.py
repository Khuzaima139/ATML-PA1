import torch
from torch.utils.data import DataLoader

from common.seed import get_device, set_seed
from task1.data.stl10 import ROOT, get_sets
from task1.models.backbones import clip_text_features, load_backbones

CACHE = ROOT / "task1/cache"


@torch.no_grad()
def extract(backbone, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    set_seed()
    device = get_device()
    sets, classes = get_sets()
    backbones, clip_model, tokenizer = load_backbones(device)
    CACHE.mkdir(parents=True, exist_ok=True)

    for name, backbone in backbones.items():
        for split_name, dataset in sets.items():
            path = CACHE / f"{name}_{split_name}_clean.pt"
            if path.exists():
                print(f"loaded {path.name}")
                continue
            feats, labels = extract(backbone, dataset, device)
            torch.save({"feats": feats, "labels": labels}, path)
            print(f"extracted {path.name}: {tuple(feats.shape)}")

    text_path = CACHE / "clip_text.pt"
    if not text_path.exists():
        text = clip_text_features(clip_model, tokenizer, classes, device).cpu()
        torch.save({"text": text, "classes": classes}, text_path)
        print(f"extracted {text_path.name}: {tuple(text.shape)}")


if __name__ == "__main__":
    main()