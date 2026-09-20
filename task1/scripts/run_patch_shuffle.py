import json

import matplotlib.pyplot as plt
import torch

from common.seed import get_device, set_seed
from task1.analysis.evaluate_bias import BACKBONES, consistency, get_probs, load_heads, summarize
from task1.data.stl10 import ROOT, cfg, get_sets
from task1.data.transforms import make_permutations, patch_shuffle
from task1.models.backbones import load_backbones
from task1.scripts.extract_features import extract

CACHE = ROOT / "task1/cache"
CKPT = ROOT / "checkpoints/task1"
RESULTS = ROOT / "task1/results"
FIGURES = ROOT / "report/figures"


def main():
    set_seed()
    device = get_device()
    sets, classes = get_sets()
    eval_set = sets["eval"]
    grid = cfg["patch_shuffle"]["grid"]

    perms = make_permutations(len(eval_set), grid * grid, cfg["seed"])
    counter = {"i": 0}

    def transform(x):
        start = counter["i"]
        batch_perms = perms[start:start + len(x)]
        counter["i"] = start + len(x)
        return patch_shuffle(x, batch_perms, grid)

    backbones, clip_model, _ = load_backbones(device)
    logit_scale = clip_model.logit_scale.exp().item()
    text_feats = torch.load(CACHE / "clip_text.pt")["text"]

    feats = {}
    for name, backbone in backbones.items():
        path = CACHE / f"{name}_eval_shuffle.pt"
        if path.exists():
            feats[name] = torch.load(path)["feats"]
            print(f"loaded {path.name}")
        else:
            counter["i"] = 0
            f, y = extract(backbone, eval_set, device, transform)
            torch.save({"feats": f, "labels": y}, path)
            feats[name] = f
            print(f"extracted {path.name}")

    clean_feats = {n: torch.load(CACHE / f"{n}_eval_clean.pt")["feats"] for n in BACKBONES}
    heads = load_heads(CKPT, clean_feats)
    with open(RESULTS / "clean_baseline.json") as f:
        clean = json.load(f)["models"]
    with open(RESULTS / "predictions_clean.json") as f:
        clean_preds = json.load(f)
    labels = torch.tensor(clean_preds["labels"])

    probs = get_probs(heads, text_feats, logit_scale, feats)
    results, predictions = {}, {"labels": labels.tolist()}
    print(f"\n{'model':<16}{'acc':>8}{'d_acc':>8}{'consist':>9}{'conf':>8}{'conf_r':>8}")
    for model, p in probs.items():
        m = summarize(p, labels)
        preds = p.argmax(dim=1)
        m["delta_acc"] = m["acc"] - clean[model]["acc"]
        m["delta_macro_f1"] = m["macro_f1"] - clean[model]["macro_f1"]
        m["conf_ratio"] = m["mean_max_conf"] / clean[model]["mean_max_conf"]
        m["consistency"] = consistency(preds, torch.tensor(clean_preds[model]))
        results[model] = m
        predictions[model] = preds.tolist()
        print(f"{model:<16}{m['acc']:>8.2f}{m['delta_acc']:>8.2f}{m['consistency']:>9.2f}"
              f"{m['mean_max_conf']:>8.2f}{m['conf_ratio']:>8.2f}")

    with open(RESULTS / "patch_shuffle.json", "w") as f:
        json.dump({"grid": grid, "models": results}, f, indent=2)
    with open(RESULTS / "predictions_shuffle.json", "w") as f:
        json.dump(predictions, f)

    idx = [labels.tolist().index(c) for c in range(5)]
    images = torch.stack([eval_set[i][0] for i in idx])
    shuffled = patch_shuffle(images, perms[idx], grid)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for c in range(5):
        for r, img in enumerate([images[c], shuffled[c]]):
            ax = axes[r, c]
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(["clean", "shuffled"][r])
            if r == 0:
                ax.set_title(classes[c])
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "task1_patch_shuffle_examples.png", dpi=150)
    plt.close(fig)
    print("saved report/figures/task1_patch_shuffle_examples.png")


if __name__ == "__main__":
    main()