import json

import torch

from task1.analysis.evaluate_bias import BACKBONES, get_probs, load_heads, summarize
from task1.data.stl10 import ROOT
from task1.models.backbones import clip_logit_scale

CACHE = ROOT / "task1/cache"
CKPT = ROOT / "checkpoints/task1"
RESULTS = ROOT / "task1/results"


def main():
    feats = {}
    for name in BACKBONES:
        d = torch.load(CACHE / f"{name}_eval_clean.pt")
        feats[name] = d["feats"]
        labels = d["labels"]
    text_feats = torch.load(CACHE / "clip_text.pt")["text"]
    logit_scale = clip_logit_scale()

    heads = load_heads(CKPT, feats)
    probs = get_probs(heads, text_feats, logit_scale, feats)

    results = {"logit_scale": logit_scale, "models": {}}
    predictions = {"labels": labels.tolist()}
    print(f"{'model':<16}{'acc':>8}{'macro_f1':>10}{'max_conf':>10}")
    for model, p in probs.items():
        m = summarize(p, labels)
        results["models"][model] = m
        predictions[model] = p.argmax(dim=1).tolist()
        print(f"{model:<16}{m['acc']:>8.2f}{m['macro_f1']:>10.2f}{m['mean_max_conf']:>10.2f}")

    with open(RESULTS / "clean_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS / "predictions_clean.json", "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()