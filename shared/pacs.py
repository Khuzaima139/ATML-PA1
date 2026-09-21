from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from shared.pacs_protocol import DATA, SOURCES, TARGET, label_of, load_splits

_weights = ResNet18_Weights.IMAGENET1K_V1.transforms()
_normalize = transforms.Normalize(mean=_weights.mean, std=_weights.std)

TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _normalize,
])

EVAL_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _normalize,
])


class PACS(Dataset):
    def __init__(self, files, transform, with_labels=True):
        self.files = files
        self.transform = transform
        self.with_labels = with_labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = self.transform(Image.open(DATA / self.files[i]).convert("RGB"))
        if not self.with_labels:
            return x
        return x, label_of(self.files[i])


def source_dataset(domain, split, train):
    return PACS(load_splits()[domain][split], TRAIN_TF if train else EVAL_TF)


def target_dataset(train):
    return PACS(load_splits()[TARGET]["all"], TRAIN_TF if train else EVAL_TF, with_labels=False)


def target_labeled_dataset():
    # only for final evaluation, after every model and setting is fixed
    return PACS(load_splits()[TARGET]["all"], EVAL_TF)


if __name__ == "__main__":
    for d in SOURCES:
        for split, train in [("train", True), ("val", False)]:
            ds = source_dataset(d, split, train)
            x, y = ds[0]
            print(d, split, len(ds), tuple(x.shape), "label", y)
    t = target_dataset(train=True)
    print(TARGET, "unlabeled", len(t), tuple(t[0].shape))