import json

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import STL10
from torchvision.utils import save_image

from common.seed import get_device, set_seed
from task1.data.stl10 import BASE_TRANSFORM, ROOT, cfg, get_sets, load_style_pool
from task1.models.adain import load_adain, style_transfer

OUT = ROOT / "task1/cache/cue_conflicts"
REVIEW = ROOT / "task1/cache/cue_review"
MANIFEST = ROOT / "task1/data/cue_conflicts_generated.json"


def save_review_grid(images, shape_cls, texture_cls, path):
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    for k, (ax, img) in enumerate(zip(axes.flat, images)):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(str(k))
        ax.axis("off")
    fig.suptitle(f"shape: {shape_cls}    texture: {texture_cls}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    set_seed()
    device = get_device()
    cc = cfg["cue_conflict"]
    n = cc["generate_per_direction"]

    sets, classes = get_sets()
    eval_set = sets["eval"]
    eval_labels = np.array(eval_set.dataset.labels[eval_set.indices])
    train_full = STL10(ROOT / cfg["data_root"], split="train", transform=BASE_TRANSFORM)
    pool = load_style_pool()
    encoder, decoder = load_adain(device)

    rng = np.random.default_rng(cfg["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    manifest = []

    for a, b in cc["pairs"]:
        for shape_cls, texture_cls in [(a, b), (b, a)]:
            direction = f"{shape_cls}+{texture_cls}"
            candidates = np.where(eval_labels == classes.index(shape_cls))[0]
            content_pos = rng.choice(candidates, size=n, replace=False)
            style_idx = rng.permutation(np.repeat(pool[texture_cls], n // len(pool[texture_cls])))

            images = []
            for k, (cp, si) in enumerate(zip(content_pos, style_idx)):
                content = eval_set[int(cp)][0][None].to(device)
                style = train_full[int(si)][0][None].to(device)
                out = style_transfer(encoder, decoder, content, style, cc["alpha"])[0].cpu()
                image_id = f"{direction}_{k:02d}"
                save_image(out, OUT / f"{image_id}.png")
                images.append(out)
                manifest.append({
                    "id": image_id,
                    "direction": direction,
                    "shape": shape_cls,
                    "texture": texture_cls,
                    "eval_pos": int(cp),
                    "style_train_idx": int(si),
                })

            save_review_grid(images, shape_cls, texture_cls, REVIEW / f"{direction}.png")
            print(f"generated {direction}: {len(images)}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"total {len(manifest)}, manifest saved to {MANIFEST}")


if __name__ == "__main__":
    main()