import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from torchvision import datasets

from task4.data.cifar10 import DATA_DIR
from task4.data.cifar100_unknowns import unknowns
from task4.evaluation.failure_analysis import accepted_by_class, most_confident_accepted
from task4.evaluation.metrics import auroc
from task4.evaluation.thresholds import accept_rate, threshold
from task4.extract_outputs import load_outputs
from task4.scores.energy import energy
from task4.scores.mahalanobis import fit, mahalanobis
from task4.scores.mls import mls
from task4.scores.msp import msp
from task4.scores.placeholder import placeholder

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
OUT_PATH = ROOT / "task4" / "results" / "osr_eval.json"
SPLITS = ["val", "test", "near", "far"]
LOGIT_SCORES = {"MSP": msp, "MLS": mls, "Energy": energy}


def unknownness(cache, name):
    if name == "Mahalanobis":
        means, var = fit(cache["train_feat"], cache["train_labels"])
        return {s: mahalanobis(cache[f"{s}_feat"], means, var) for s in SPLITS}
    if name == "Placeholder":
        return {s: placeholder(cache[f"{s}_logits"], cache[f"{s}_dummy"]) for s in SPLITS}
    return {s: LOGIT_SCORES[name](cache[f"{s}_logits"]) for s in SPLITS}


def evaluate(cache, u):
    tau = threshold(u["val"])
    unk = np.r_[u["near"], u["far"]]
    return {
        "csa": 100 * float((cache["test_logits"].argmax(1) == cache["test_labels"]).mean()),
        "auroc_near": auroc(u["test"], u["near"]),
        "auroc_far": auroc(u["test"], u["far"]),
        "auroc_all": auroc(u["test"], unk),
        "tau": tau,
        "test_accept": accept_rate(u["test"], tau),
        "near_reject": 100 - accept_rate(u["near"], tau),
        "far_reject": 100 - accept_rate(u["far"], tau),
        "all_reject": 100 - accept_rate(unk, tau),
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'model':8} {'score':12} {'CSA':>6} {'AUC near':>9} {'AUC far':>8} {'AUC all':>8} "
          f"{'test acc':>9} {'rej near':>9} {'rej far':>8} {'rej all':>8}")
    for r in rows:
        print(f"{r['model']:8} {r['score']:12} {r['csa']:6.2f} {r['auroc_near']:9.2f} {r['auroc_far']:8.2f} "
              f"{r['auroc_all']:8.2f} {r['test_accept']:9.2f} {r['near_reject']:9.2f} {r['far_reject']:8.2f} "
              f"{r['all_reject']:8.2f}")


def roc_figure(us):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, ["MSP", "MLS", "Mahalanobis"]):
        u = us[("vanilla", name)]
        tau = threshold(u["val"])
        for group, color in [("near", "tab:red"), ("far", "tab:blue")]:
            y = np.r_[np.ones(len(u["test"])), np.zeros(len(u[group]))]
            fpr, tpr, _ = roc_curve(y, -np.r_[u["test"], u[group]])
            ax.plot(100 * fpr, 100 * tpr, color=color, label=f"{group} (AUROC {auroc(u['test'], u[group]):.1f})")
            ax.plot(accept_rate(u[group], tau), accept_rate(u["test"], tau), "o", color=color)
        ax.plot([0, 100], [0, 100], ":", color="grey")
        ax.set_title(f"Vanilla, {name}")
        ax.set_xlabel("unknowns accepted (%)")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("CIFAR-10 test accepted (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_roc_vanilla.png", dpi=150)
    plt.close(fig)


def failure_figure(cases, tau):
    fig, axes = plt.subplots(2, 8, figsize=(14, 4.6))
    for row, group in zip(axes, ["near", "far"]):
        ds = unknowns(group, None)
        for ax in row:
            ax.axis("off")
        for ax, c in zip(row, [c for c in cases if c["group"] == group]):
            ax.imshow(ds[c["position"]][0])
            ax.set_title(f"{c['unknown_class']}\nas {c['predicted']}\nu = {c['score']:.2f}", fontsize=8)
    fig.suptitle(f"Most confidently accepted image per unknown class, Vanilla MLS (tau = {tau:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task4_failures_vanilla_mls.png", dpi=150)
    plt.close(fig)


def main():
    caches = {run: load_outputs(run) for run in ["vanilla", "gcsc", "proser"]}
    print("loaded cached outputs: vanilla, gcsc, proser")

    table1_keys = [("vanilla", s) for s in ["MSP", "MLS", "Energy", "Mahalanobis"]]
    table2_keys = [("vanilla", "MLS"), ("gcsc", "MLS"), ("proser", "MLS"), ("proser", "Placeholder")]
    us = {k: unknownness(caches[k[0]], k[1]) for k in dict.fromkeys(table1_keys + table2_keys)}
    table1 = [{"model": m, "score": s, **evaluate(caches[m], us[(m, s)])} for m, s in table1_keys]
    table2 = [{"model": m, "score": s, **evaluate(caches[m], us[(m, s)])} for m, s in table2_keys]
    print_table("Table 1: post-hoc scores on Vanilla", table1)
    print_table("Table 2: trained models", table2)

    v, u = caches["vanilla"], us[("vanilla", "MLS")]
    tau = threshold(u["val"])
    c10 = datasets.CIFAR10(DATA_DIR, train=False).classes
    c100 = datasets.CIFAR100(DATA_DIR, train=False).classes
    per_class, cases = {}, []
    for group in ["near", "far"]:
        pred = v[f"{group}_logits"].argmax(1)
        per_class[group] = accepted_by_class(u[group], tau, pred, v[f"{group}_labels"], c100, c10)
        cases += most_confident_accepted(u[group], tau, pred, v[f"{group}_labels"], c100, c10, group)
    print(f"\nVanilla MLS threshold tau = {tau:.3f}; accepted unknowns per class")
    for group in ["near", "far"]:
        for name, r in per_class[group].items():
            top = ", ".join(f"{k} {n}" for k, n in list(r["predicted"].items())[:3])
            print(f"{group:4} {name:13} {r['accepted']:3d}/{r['of']}  {top}")
    print("\nMost confidently accepted per class")
    for c in cases:
        print(f"{c['group']:4} {c['unknown_class']:13} as {c['predicted']:10} u = {c['score']:.3f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    roc_figure(us)
    failure_figure(cases, tau)
    OUT_PATH.write_text(json.dumps({"table1": table1, "table2": table2, "vanilla_mls_tau": tau,
                                    "accepted_by_class": per_class, "failure_cases": cases}, indent=2))
    print(f"\ncreated {OUT_PATH}, report/figures/task4_roc_vanilla.png, report/figures/task4_failures_vanilla_mls.png")


if __name__ == "__main__":
    main()