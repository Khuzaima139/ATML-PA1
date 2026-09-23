import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task3.models.backbone import ROOT

RESULTS = ROOT / "task3" / "results"
FIGURES = ROOT / "report" / "figures"
PANELS = [("ERM (Task 2 Source-only)", ROOT / "task2" / "results" / "source_only_train.json", None, None),
          ("DAN-DG (λ_DG = 1)", RESULTS / "dan_dg_train.json", "mmd", "MMD² (mean over pairs)"),
          ("SAM (ρ = 0.05)", RESULTS / "sam_train.json", "cls_perturbed", "loss at θ + ε")]
LAMBDAS = [(0.1, "dan_dg_lambda0.1"), (1.0, "dan_dg"), (10.0, "dan_dg_lambda10")]


def training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (title, path, extra, extra_label) in zip(axes, PANELS):
        r = json.loads(path.read_text())
        epochs = [h["epoch"] for h in r["history"]]
        ax.plot(epochs, [h["train"]["cls"] for h in r["history"]], color="C0", label="class loss")
        if extra:
            ax.plot(epochs, [h["train"][extra] for h in r["history"]], color="C1", label=extra_label)
        ax.set_yscale("log")
        ax.axvline(r["best_epoch"], color="grey", lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax2 = ax.twinx()
        ax2.plot(epochs, [h["val"]["mean"]["macro_f1"] for h in r["history"]],
                 color="black", ls="--", label="source-val F1")
        ax2.set_ylim(0, 100)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    axes[0].set_ylabel("training loss (log scale)")
    ax2.set_ylabel("mean source-val macro-F1 (%)")
    fig.tight_layout()
    out = FIGURES / "task3_training_curves.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out.name}")


def lambda_study():
    ev = json.loads((RESULTS / "final_eval.json").read_text())
    xs = [lam for lam, _ in LAMBDAS]
    series = {
        "mean source-val macro-F1": lambda r: r["source_val"]["mean"]["macro_f1"],
        "source-domain separability": lambda r: r["source_separability"],
        "Sketch accuracy": lambda r: r["sketch"]["acc"],
        "Sketch macro-F1": lambda r: r["sketch"]["macro_f1"],
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for label, get in series.items():
        ax.plot(xs, [get(ev[name]) for _, name in LAMBDAS], marker="o", label=label)
    ax.axhline(100 / 3, color="grey", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(xs, ["0.1", "1", "10"])
    ax.set_xlabel("λ_DG")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "task3_lambda_study.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    training_curves()
    lambda_study()