import json
from functools import lru_cache
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "PACS"
SPLIT_FILE = ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"
SEED = 6304
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
SOURCES = ["photo", "art_painting", "cartoon"]
TARGET = "sketch"


def list_domain(domain):
    files = (DATA / domain).rglob("*")
    return sorted(p.relative_to(DATA).as_posix() for p in files if p.suffix.lower() in {".jpg", ".png"})


def label_of(path):
    return CLASSES.index(path.split("/")[1])


def make_splits():
    splits = {"seed": SEED, "classes": CLASSES, "sources": SOURCES, "target": TARGET}
    for d in SOURCES:
        files = list_domain(d)
        labels = [label_of(f) for f in files]
        train, val = train_test_split(files, test_size=0.2, stratify=labels, random_state=SEED)
        splits[d] = {"train": sorted(train), "val": sorted(val)}
    splits[TARGET] = {"all": list_domain(TARGET)}
    return splits


@lru_cache(maxsize=None)
def load_splits():
    if SPLIT_FILE.exists():
        print(f"loaded {SPLIT_FILE.name}")
        return json.loads(SPLIT_FILE.read_text())
    splits = make_splits()
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(splits, indent=1))
    print(f"created {SPLIT_FILE.name}")
    return splits


if __name__ == "__main__":
    s = load_splits()
    for d in SOURCES:
        train, val = s[d]["train"], s[d]["val"]
        assert not set(train) & set(val)
        val_counts = [sum(label_of(f) == c for f in val) for c in range(len(CLASSES))]
        print(d, "train", len(train), "val", len(val), "val per class", val_counts)
    print(TARGET, "all", len(s[TARGET]["all"]), "(unlabeled during adaptation)")