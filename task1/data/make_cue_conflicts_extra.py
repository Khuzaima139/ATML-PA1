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


def main():
    set_seed()
    device = get_device()
    cc = cfg["cue_conflict"]
    with open(MANIFEST) as f:
        manifest = json.load(f)
    if len(manifest) > 10 * cc["generate_per_direction"]:
        print("extra batch already generated, stopping")
        return

    sets, classes = get_sets()
    eval_set = sets["eval"]
    eval_labels = np.array(eval_set.dataset.labels[eval_set.indices])
    train_full = STL10(ROOT / cfg["data_root"], split="train", transform=BASE_TRANSFORM)
    pool = load_style_pool()
    encoder, decoder = load_adain(device)
    rng = np.random.default_rng(cfg["seed"] + 1)

    for a, b in cc["pairs"]:
        for shape_cls, texture_cls in [(a, b), (b, a)]:
            direction = f"{shape_cls}+{texture_cls}"
            used = {m["eval_pos"] for m in manifest if m["direction"] == direction}
            candidates = [int(p) for p in np.where(eval_labels == classes.index(shape_cls))[0] if p not in used]
            style_idx = rng.permutation(np.repeat(pool[texture_cls], len(candidates) // len(pool[texture_cls])))

            images, numbers = [], []
            for k, (cp, si) in enumerate(zip(candidates, style_idx), start=len(used)):
                content = eval_set[cp][0][None].to(device)
                style = train_full[int(si)][0][None].to(device)
                out = style_transfer(encoder, decoder, content, style, cc["alpha"])[0].cpu()
                image_id = f"{direction}_{k:02d}"
                save_image(out, OUT / f"{image_id}.png")
                images.append(out)
                numbers.append(k)
                manifest.append({
                    "id": image_id,
                    "direction": direction,
                    "shape": shape_cls,
                    "texture": texture_cls,
                    "eval_pos": cp,
                    "style_train_idx": int(si),
                })

            fig, axes = plt.subplots(4, 5, figsize=(10, 8.5))
            for ax, img, k in zip(axes.flat, images, numbers):
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.set_title(str(k))
                ax.axis("off")
            fig.suptitle(f"EXTRA  shape: {shape_cls}    texture: {texture_cls}")
            fig.tight_layout()
            fig.savefig(REVIEW / f"{direction}_extra.png", dpi=100)
            plt.close(fig)
            print(f"generated extra {direction}: {len(images)}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"total {len(manifest)}")


if __name__ == "__main__":
    main()