from collections import Counter
from pathlib import Path

from datasets import Image, load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "PACS"
REVISION = "394113073258ead631f617d2e13bb377c0715c4b"
SOURCES = ["photo", "art_painting", "cartoon"]
TARGET = "sketch"
N_TOTAL = 9991


def image_files():
    return [p for p in OUT.rglob("*") if p.suffix.lower() in {".jpg", ".png"}]


def main():
    if len(image_files()) == N_TOTAL:
        print(f"loaded {OUT} ({N_TOTAL} images already present)")
    else:
        ds = load_dataset("flwrlabs/pacs", split="train", revision=REVISION)
        ds = ds.cast_column("image", Image(decode=False))
        names = ds.features["label"].names
        for domain, label, img in zip(ds["domain"], ds["label"], ds["image"]):
            path = OUT / domain / names[label] / img["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(img["bytes"])
        print(f"extracted {len(ds)} images to {OUT}")

    files = image_files()
    counts = Counter(p.relative_to(OUT).parts[:2] for p in files)
    for d in SOURCES:
        per_class = {c: n for (dd, c), n in sorted(counts.items()) if dd == d}
        print(d, sum(per_class.values()), per_class)
    n_target = sum(n for (dd, _), n in counts.items() if dd == TARGET)
    print(TARGET, n_target, "(class breakdown withheld until final analysis)")
    print("total", len(files))


if __name__ == "__main__":
    main()