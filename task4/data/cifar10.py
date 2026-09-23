import json
from pathlib import Path

from torch.utils.data import Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "cifar"
SPLIT_FILE = ROOT / "task4" / "data" / "splits" / "cifar10_seed6304.json"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def train_transform(randaugment=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if randaugment:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_split():
    return json.loads(SPLIT_FILE.read_text())


def cifar10(split, transform):
    if split == "test":
        return datasets.CIFAR10(DATA_DIR, train=False, transform=transform, download=True)
    full = datasets.CIFAR10(DATA_DIR, train=True, transform=transform, download=True)
    return Subset(full, load_split()[split])


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        ds = cifar10(split, eval_transform())
        x, y = ds[0]
        print(f"{split}: {len(ds)} images, sample shape {tuple(x.shape)}, label {y}")
    print("vanilla train transform:", train_transform())
    print("gcsc train transform:", train_transform(randaugment=True))