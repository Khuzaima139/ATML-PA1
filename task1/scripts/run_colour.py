import json

import matplotlib.pyplot as plt
import torch

from common.seed import get_device, set_seed
from task1.analysis.evaluate_bias import consistency, get_probs, load_heads, summarize
from task1.data.stl10 import ROOT, cfg, get_sets
from task1.data.transforms import grayscale, hue_rotate
from task1.models.backbones import load_backbones
from task1.scripts.extract_features import extract

CACHE = ROOT / "task1/cache"
CKPT = ROOT / "checkpoints/task1"
RESULTS = ROOT / "task1/results"
FIGURES = ROOT / "report/figures"


def get_features(tag, transform, backbones, dataset, device):
    feats = {}
    for name, backbone in backbones.items():
        path = CACHE / f"{name}_eval_{tag}.pt"
        if path.exists():
            feats[name] = torch.load(path)["feats"]
            print(f"loaded {path.name}")
        else:
            f, y = extract(backbone, dataset, device, transform)
            torch.save({"feats": f, "labels": y}, path)
            feats[name] = f
            print(f"extracted {path.name}")
    return feats


def save_examples(dataset, labels, classes, conditions, path, n=5):
    idx = [labels.index(c) for c in range(n)]
    images = torch.stack([dataset[i][0] for i in idx])
    fig, axes = plt.subplots(len(conditions), n, figsize=(2 * n, 2 * len(conditions)))
    for r, (name, transform) in enumerate(conditions.items()):
        shown = transform(images) if transform else images
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(shown[c].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name)
            if r == 0:
                ax.set_title(classes[c])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    set_seed()
    device = get_device()
    sets, classes = get_sets()
    eval_set = sets["eval"]

    backbones, clip_model, _ = load_backbones(device)
    logit_scale = clip_model.logit_scale.exp().item()
    text_feats = torch.load(CACHE / "clip_text.pt")["text"]

    with open(RESULTS / "clean_baseline.json") as f:
        clean = json.load(f)["models"]
    with open(RESULTS / "predictions_clean.json") as f:
        clean_preds = json.load(f)
    labels = torch.tensor(clean_preds["labels"])

    hue_factor = cfg["colour"]["hue_factor"]
    interventions = {
        "grayscale": grayscale,
        "hue180": lambda x: hue_rotate(x, hue_factor),
    }

    results = {"hue_factor": hue_factor, "interventions": {}}
    predictions = {"labels": labels.tolist()}

    for tag, transform in interventions.items():
        feats = get_features(tag, transform, backbones, eval_set, device)
        heads = load_heads(CKPT, feats)
        probs = get_probs(heads, text_feats, logit_scale, feats)
        results["interventions"][tag] = {}
        predictions[tag] = {}

        print(f"\n{tag}")
        print(f"{'model':<16}{'acc':>8}{'d_acc':>8}{'consist':>9}{'conf':>8}{'conf_r':>8}")
        for model, p in probs.items():
            m = summarize(p, labels)
            preds = p.argmax(dim=1)
            m["delta_acc"] = m["acc"] - clean[model]["acc"]
            m["delta_macro_f1"] = m["macro_f1"] - clean[model]["macro_f1"]
            m["conf_ratio"] = m["mean_max_conf"] / clean[model]["mean_max_conf"]
            m["consistency"] = consistency(preds, torch.tensor(clean_preds[model]))
            results["interventions"][tag][model] = m
            predictions[tag][model] = preds.tolist()
            print(f"{model:<16}{m['acc']:>8.2f}{m['delta_acc']:>8.2f}{m['consistency']:>9.2f}"
                  f"{m['mean_max_conf']:>8.2f}{m['conf_ratio']:>8.2f}")

    with open(RESULTS / "colour.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS / "predictions_colour.json", "w") as f:
        json.dump(predictions, f)

    FIGURES.mkdir(parents=True, exist_ok=True)
    conditions = {"clean": None, **interventions}
    save_examples(eval_set, labels.tolist(), classes, conditions, FIGURES / "task1_colour_examples.png")


if __name__ == "__main__":
    main()