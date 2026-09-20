import json

import matplotlib.pyplot as plt
from PIL import Image

from task1.data.stl10 import ROOT

RESULTS = ROOT / "task1/results"
FINAL = ROOT / "task1/cache/cue_conflicts_final"
FIGURES = ROOT / "report/figures"

MODELS = ["resnet50_head", "vit_b16_head", "clip_b32_head", "clip_zeroshot"]


def find_first(condition, manifest, used_directions):
    # first pass: prefer an example from a direction not already used
    for i, m in enumerate(manifest):
        if condition(i) and m["direction"] not in used_directions:
            return i, m
    # fallback: allow a repeat direction if nothing else matches
    for i, m in enumerate(manifest):
        if condition(i):
            return i, m
    return None


def main():
    with open(RESULTS / "cue_conflict.json") as f:
        cue = json.load(f)
    with open(RESULTS / "cue_conflicts_final.json") as f:
        manifest = json.load(f)["images"]

    conditions = [
        ("all agree: shape", lambda i: all(cue[model]["decisions"][i] == "shape" for model in MODELS)),
        ("all agree: texture", lambda i: all(cue[model]["decisions"][i] == "texture" for model in MODELS)),
        ("ResNet disagrees with CLIP zero-shot",
         lambda i: cue["resnet50_head"]["decisions"][i] != "shape" and cue["clip_zeroshot"]["decisions"][i] == "shape"),
        ("all say other", lambda i: all(cue[model]["decisions"][i] == "other" for model in MODELS)),
        ("CLIP head vs zero-shot disagree",
         lambda i: cue["clip_b32_head"]["decisions"][i] != cue["clip_zeroshot"]["decisions"][i]),
    ]

    rows, used_directions = [], set()
    for title, condition in conditions:
        found = find_first(condition, manifest, used_directions)
        if found is None:
            print(f"no match for: {title}")
            continue
        i, m = found
        used_directions.add(m["direction"])
        rows.append((title, i, m))

    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 4))
    if len(rows) == 1:
        axes = [axes]
    for ax, (title, i, m) in zip(axes, rows):
        img = Image.open(FINAL / f"{m['id']}.png")
        ax.imshow(img)
        ax.axis("off")
        caption = f"{title}\nshape={m['shape']}  texture={m['texture']}\n"
        caption += "\n".join(f"{model}: {cue[model]['decisions'][i]}" for model in MODELS)
        ax.set_title(caption, fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "task1_cue_failure_cases.png", dpi=150)
    plt.close(fig)

    with open(RESULTS / "cue_failure_cases.json", "w") as f:
        json.dump(
            [{"title": t, "index": i, **m, "decisions": {mo: cue[mo]["decisions"][i] for mo in MODELS}}
             for t, i, m in rows],
            f, indent=2,
        )
    print(f"saved {len(rows)} examples to report/figures/task1_cue_failure_cases.png")


if __name__ == "__main__":
    main()