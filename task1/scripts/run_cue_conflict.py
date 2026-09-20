import json

import torch

from task1.analysis.evaluate_bias import BACKBONES, get_probs, load_heads
from task1.analysis.shape_bias import classify_predictions, shape_bias_and_coverage
from task1.data.stl10 import ROOT
from task1.models.backbones import clip_logit_scale

CACHE = ROOT / "task1/cache"
CKPT = ROOT / "checkpoints/task1"
RESULTS = ROOT / "task1/results"


def main():
    with open(RESULTS / "cue_conflicts_final.json") as f:
        final = json.load(f)
    manifest = final["images"]
    classes = sorted({m["shape"] for m in manifest} | {m["texture"] for m in manifest})

    feats = {}
    for name in BACKBONES:
        d = torch.load(CACHE / f"{name}_cue_conflict.pt")
        feats[name] = d["feats"]
    text_feats = torch.load(CACHE / "clip_text.pt")["text"]
    logit_scale = clip_logit_scale()

    heads = load_heads(CKPT, feats)
    probs = get_probs(heads, text_feats, logit_scale, feats)

    results = {}
    print(f"{'model':<16}{'shape_bias':>11}{'coverage':>10}{'n_shape':>9}{'n_texture':>10}{'n_other':>9}")
    for model, p in probs.items():
        preds = p.argmax(dim=1)
        decisions = classify_predictions(preds, manifest, classes)
        m = shape_bias_and_coverage(decisions)
        m["mean_max_conf"] = 100 * p.max(dim=1).values.mean().item()
        m["decisions"] = decisions
        results[model] = m
        print(f"{model:<16}{m['shape_bias']:>11.2f}{m['coverage']:>10.2f}"
              f"{m['n_shape']:>9}{m['n_texture']:>10}{m['n_other']:>9}")

    with open(RESULTS / "cue_conflict.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()