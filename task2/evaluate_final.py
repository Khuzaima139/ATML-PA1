import json

import torch
from torch.utils.data import DataLoader

from common.metrics import acc_f1
from common.seed import get_device
from shared.pacs import source_dataset, target_labeled_dataset
from shared.pacs_protocol import CLASSES, SOURCES
from task2.evaluation.class_analysis import dominant_confusion, per_class
from task2.evaluation.domain_separability import domain_separability, separability_sets
from task2.evaluation.metrics import predict, source_val_metrics
from task2.models.backbone import Backbone
from task2.models.classifier_head import ClassifierHead
from task2.train import ROOT

MAIN = ["source_only", "dan", "dann_l2", "cdan_l2"]
STUDY = ["dan_lambda0.1", "dan_lambda10"]
SPECIFIED = ["dann", "cdan"]


def load(run, device):
    backbone = Backbone().to(device)
    head = ClassifierHead(backbone.out_dim, len(CLASSES)).to(device)
    ckpt = torch.load(ROOT / "checkpoints" / "task2" / f"{run}.pt", map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    return backbone, head, ckpt["epoch"]


def main():
    device = get_device()
    val_loaders = {d: DataLoader(source_dataset(d, "val", train=False), batch_size=64) for d in SOURCES}
    tgt_loader = DataLoader(target_labeled_dataset(), batch_size=64)
    sep_src, sep_tgt = separability_sets()

    results = {}
    for run in MAIN + STUDY + SPECIFIED:
        backbone, head, epoch = load(run, device)
        val = source_val_metrics(backbone, head, val_loaders, device)
        y, pred = predict(backbone, head, tgt_loader, device)
        results[run] = {
            "epoch": epoch,
            "source_val": val,
            "target": acc_f1(y, pred),
            "target_classes": per_class(y, pred),
            "domain_separability": domain_separability(backbone, sep_src, sep_tgt, device),
        }
        logged = json.loads((ROOT / "task2" / "results" / f"{run}_train.json").read_text())["best_val"]["mean"]["macro_f1"]
        print(f"evaluated {run} (epoch {epoch}); source-val mean F1 {val['mean']['macro_f1']:.2f}, training log {logged:.2f}")

    base = results["source_only"]
    for r in results.values():
        r["target_acc_change"] = r["target"]["acc"] - base["target"]["acc"]

    print(f"\n{'run':14s} " + " ".join(f"{d[:5]:>11s}" for d in SOURCES)
          + f" {'mean acc/F1':>12s} {'sketch acc/F1':>14s} {'d acc':>6s} {'sep':>6s}")
    for run in MAIN + STUDY + SPECIFIED:
        r = results[run]
        doms = " ".join(f"{r['source_val'][d]['acc']:5.1f}/{r['source_val'][d]['macro_f1']:5.1f}" for d in SOURCES)
        m, t = r["source_val"]["mean"], r["target"]
        print(f"{run:14s} {doms} {m['acc']:5.1f}/{m['macro_f1']:5.1f}  {t['acc']:6.1f}/{t['macro_f1']:5.1f}"
              f" {r['target_acc_change']:+6.1f} {r['domain_separability']:6.1f}")

    print("\nper-class Sketch accuracy (change vs source_only), dominant confusion")
    base_acc, base_cm = base["target_classes"]["acc"], base["target_classes"]["confusion"]
    for c, name in enumerate(CLASSES):
        wrong, n = dominant_confusion(base_cm, c)
        print(f"  source_only {name:9s} {base_acc[name]:5.1f}  -> {wrong} ({n})")
    for run in MAIN[1:] + STUDY + SPECIFIED:
        print(f" {run}")
        acc, cm = results[run]["target_classes"]["acc"], results[run]["target_classes"]["confusion"]
        for c, name in enumerate(CLASSES):
            wrong, n = dominant_confusion(cm, c)
            print(f"  {name:9s} {acc[name]:5.1f} ({acc[name] - base_acc[name]:+5.1f})  -> {wrong} ({n})")

    out = ROOT / "task2" / "results" / "final_eval.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()