import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from task1.analysis.feature_similarity import cosine_stability
from task1.analysis.representation import fit_umap
from task1.data.stl10 import ROOT, cfg

CACHE = ROOT / "task1/cache"
RESULTS = ROOT / "task1/results"
FIGURES = ROOT / "report/figures"

BACKBONES = ["resnet50", "vit_b16", "clip_b32"]
DIRECTIONS = ["right", "left", "down", "up"]


def main():
    seed = cfg["seed"]
    rep = cfg["representation"]
    delta, vis_dir = rep["translation_delta"], rep["vis_direction"]

    with open(ROOT / cfg["split_file"]) as f:
        classes = json.load(f)["classes"]
    with open(RESULTS / "cue_conflicts_final.json") as f:
        cue_manifest = json.load(f)["images"]
    cue_pos = torch.tensor([m["eval_pos"] for m in cue_manifest])
    cue_labels = torch.tensor([classes.index(m["shape"]) for m in cue_manifest])

    stability, panels = {}, {}

    for name in BACKBONES:
        clean = torch.load(CACHE / f"{name}_eval_clean.pt")
        clean_feats, labels = clean["feats"], clean["labels"]

        gray = torch.load(CACHE / f"{name}_eval_grayscale.pt")["feats"]
        hue = torch.load(CACHE / f"{name}_eval_hue180.pt")["feats"]
        shuffle = torch.load(CACHE / f"{name}_eval_shuffle.pt")["feats"]
        cue = torch.load(CACHE / f"{name}_cue_conflict.pt")["feats"]
        trans = {d: torch.load(CACHE / f"{name}_eval_trans_d{delta}_{d}.pt")["feats"] for d in DIRECTIONS}

        stability[name] = {
            "grayscale": cosine_stability(clean_feats, gray),
            "hue180": cosine_stability(clean_feats, hue),
            "cue_conflict": cosine_stability(clean_feats[cue_pos], cue),
            f"translation_d{delta}": float(np.mean([cosine_stability(clean_feats, t) for t in trans.values()])),
            "patch_shuffle": cosine_stability(clean_feats, shuffle),
        }

        conditions = [
            ("grayscale", gray, labels),
            ("cue conflict", cue, cue_labels),
            (f"translation {delta}px", trans[vis_dir], labels),
            ("patch shuffle", shuffle, labels),
        ]
        stacked = torch.cat([clean_feats] + [f for _, f, _ in conditions]).numpy()
        emb = fit_umap(stacked, seed, rep["n_neighbors"], rep["min_dist"], rep["metric"])

        offset = len(clean_feats)
        clean_emb = emb[:offset]
        panels[name] = []
        for title, feats, cond_labels in conditions:
            panels[name].append((title, clean_emb, labels.numpy(),
                                 emb[offset:offset + len(feats)], cond_labels.numpy()))
            offset += len(feats)

    print(f"{'backbone':<12}" + "".join(f"{k:>18}" for k in stability[BACKBONES[0]]))
    for name in BACKBONES:
        print(f"{name:<12}" + "".join(f"{v:>18.2f}" for v in stability[name].values()))

    with open(RESULTS / "representation_stability.json", "w") as f:
        json.dump({"settings": rep, "cosine_stability": stability}, f, indent=2)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(BACKBONES), 4, figsize=(16, 12))
    for r, name in enumerate(BACKBONES):
        for c, (title, ce, cl, te, tl) in enumerate(panels[name]):
            ax = axes[r, c]
            ax.scatter(ce[:, 0], ce[:, 1], c=[cmap(i) for i in cl], s=6, alpha=0.45, marker="o")
            ax.scatter(te[:, 0], te[:, 1], c=[cmap(i) for i in tl], s=14, alpha=0.9, marker="x")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(name)

    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i), label=c) for i, c in enumerate(classes)]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="grey", label="clean"),
        plt.Line2D([], [], marker="x", ls="", color="grey", label="transformed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle("UMAP of clean and transformed features (one projection fitted per backbone)")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "task1_representation_umap.png", dpi=150)
    plt.close(fig)
    print("saved report/figures/task1_representation_umap.png")


if __name__ == "__main__":
    main()