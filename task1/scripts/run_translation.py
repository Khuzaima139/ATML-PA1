import json

import matplotlib.pyplot as plt
import torch

from common.seed import get_device, set_seed
from task1.analysis.evaluate_bias import consistency, get_probs, load_heads, summarize
from task1.data.stl10 import ROOT, cfg, get_sets
from task1.data.transforms import translate
from task1.models.backbones import load_backbones
from task1.scripts.extract_features import extract

CACHE = ROOT / "task1/cache"
CKPT = ROOT / "checkpoints/task1"
RESULTS = ROOT / "task1/results"
FIGURES = ROOT / "report/figures"

BACKBONES = ["resnet50", "vit_b16", "clip_b32"]
MODELS = ["resnet50_head", "vit_b16_head", "clip_b32_head", "clip_zeroshot"]
DIRECTIONS = ["right", "left", "down", "up"]


def get_features(tag, transform, backbones, dataset, device):
    feats = {}
    for name, backbone in backbones.items():
        path = CACHE / f"{name}_eval_{tag}.pt"
        if path.exists():
            feats[name] = torch.load(path)["feats"]
        else:
            f, y = extract(backbone, dataset, device, transform)
            torch.save({"feats": f, "labels": y}, path)
            feats[name] = f
    return feats


def eval_point(heads, text_feats, logit_scale, feats, labels, clean_preds):
    probs = get_probs(heads, text_feats, logit_scale, feats)
    out = {}
    for model, p in probs.items():
        m = summarize(p, labels)
        preds = p.argmax(dim=1)
        out[model] = {
            "acc": m["acc"],
            "consistency": consistency(preds, torch.tensor(clean_preds[model])),
            "conf": m["mean_max_conf"],
        }
    return out


def main():
    set_seed()
    device = get_device()
    sets, classes = get_sets()
    eval_set = sets["eval"]

    backbones, clip_model, _ = load_backbones(device)
    logit_scale = clip_model.logit_scale.exp().item()
    text_feats = torch.load(CACHE / "clip_text.pt")["text"]

    clean_feats = {n: torch.load(CACHE / f"{n}_eval_clean.pt")["feats"] for n in BACKBONES}
    heads = load_heads(CKPT, clean_feats)

    with open(RESULTS / "predictions_clean.json") as f:
        clean_preds = json.load(f)
    labels = torch.tensor(clean_preds["labels"])

    deltas = cfg["translation"]["deltas"]
    results = {"deltas": deltas, "models": {m: {"acc": [], "consistency": [], "conf": []} for m in MODELS}}

    for delta in deltas:
        if delta == 0:
            point = eval_point(heads, text_feats, logit_scale, clean_feats, labels, clean_preds)
        else:
            per_dir = {m: {"acc": [], "consistency": [], "conf": []} for m in MODELS}
            for direction in DIRECTIONS:
                tag = f"trans_d{delta}_{direction}"
                transform = lambda x, d=delta, dirn=direction: translate(x, d, dirn)
                feats = get_features(tag, transform, backbones, eval_set, device)
                p = eval_point(heads, text_feats, logit_scale, feats, labels, clean_preds)
                for model in MODELS:
                    for key in ("acc", "consistency", "conf"):
                        per_dir[model][key].append(p[model][key])
            point = {
                model: {key: sum(vals) / len(vals) for key, vals in per_dir[model].items()}
                for model in MODELS
            }

        for model in MODELS:
            for key in ("acc", "consistency", "conf"):
                results["models"][model][key].append(point[model][key])
        print(f"delta={delta:>3}  " + "  ".join(f"{m}: acc={point[m]['acc']:.2f} cons={point[m]['consistency']:.2f}" for m in MODELS))

    with open(RESULTS / "translation.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model in MODELS:
        axes[0].plot(deltas, results["models"][model]["acc"], marker="o", label=model)
        axes[1].plot(deltas, results["models"][model]["consistency"], marker="o", label=model)
    axes[0].set_xlabel("displacement (px)")
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Accuracy vs displacement")
    axes[1].set_xlabel("displacement (px)")
    axes[1].set_ylabel("consistency (%)")
    axes[1].set_title("Consistency vs displacement")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "task1_translation_curve.png", dpi=150)
    plt.close(fig)
    print("saved report/figures/task1_translation_curve.png")


if __name__ == "__main__":
    main()