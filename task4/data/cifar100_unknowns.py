from collections import Counter

from torch.utils.data import Subset
from torchvision import datasets

from task4.data.cifar10 import DATA_DIR, eval_transform

GROUPS = {
    "near": ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"],
    "far": ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"],
}


def unknowns(group, transform):
    ds = datasets.CIFAR100(DATA_DIR, train=False, transform=transform, download=True)
    wanted = {ds.class_to_idx[name] for name in GROUPS[group]}
    idx = [i for i, y in enumerate(ds.targets) if y in wanted]
    assert len(idx) == 800, len(idx)
    return Subset(ds, idx)


if __name__ == "__main__":
    for group in GROUPS:
        ds = unknowns(group, eval_transform())
        classes = ds.dataset.classes
        counts = Counter(classes[ds.dataset.targets[i]] for i in ds.indices)
        print(f"{group}: {len(ds)} images, {dict(counts)}")