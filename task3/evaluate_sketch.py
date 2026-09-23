import json

from torch.utils.data import DataLoader

from common.metrics import acc_f1
from common.seed import get_device
from shared.pacs import target_labeled_dataset
from shared.pacs_protocol import CLASSES, SOURCES
from task2.evaluation.class_analysis import dominant_confusion, per_class
from task2.evaluation.metrics import predict
from task3.models.backbone import MAIN, ROOT, STUDY, load_run
from task3.selection.source_validation import evaluate_sources, val_loaders

RESULTS = ROOT / "task3" / "results"
PAIRS = [(0.1, "dan_lambda0.1", "dan_dg_lambda0.1"), (1, "dan", "dan_dg"), (10, "dan_lambda10", "dan_dg_lambda10")]


def logged_mean_f1(run):
    if run == "erm":
        return json.loads((RESULTS / "erm_source_val.json").read_text())["val"]["mean"]["macro_f1"]
    return json.loads((RESULTS / f"{run}_train.json").read_text())["best_val"]["mean"]["macro_f1"]


def main():
    device = get_device()
    loaders = val_loaders()
    tgt = DataLoader(target_labeled_dataset(), batch_size=64)
    sep = json.loads((RESULTS / "source_separability.json").read_text())
    sharp = json.loads((RESULTS / "sharpness.json").read_text())

    results = {}
    for run in MAIN + STUDY:
        backbone, head, ckpt = load_run(run, device)
        val = evaluate_sources(backbone, head, loaders, device)
        y, pred = predict(backbone, head, tgt, device)
        results[run] = {"epoch": ckpt["epoch"], "source_val": val, "sketch": acc_f1(y, pred),
                        "sketch_classes": per_class(y, pred),
                        "source_separability": sep[run]["separability"],
                        "sharpness": sharp[run]["delta"] if run in sharp else None}
        print(f"evaluated {run} (epoch {ckpt['epoch']}); source-val mean F1 {val['mean']['macro_f1']:.2f},"
              f" training log {logged_mean_f1(run):.2f}")

    base = results["erm"]
    for r in results.values():
        r["sketch_acc_change"] = r["sketch"]["acc"] - base["sketch"]["acc"]

    print(f"\n{'run':17s} " + " ".join(f"{d[:5]:>11s}" for d in SOURCES)
          + f" {'mean':>11s} {'worst':>11s} {'sketch':>11s} {'d acc':>6s} {'sep':>5s} {'sharp':>7s}")
    for run in MAIN + STUDY:
        r = results[run]
        doms = " ".join(f"{r['source_val'][d]['acc']:5.1f}/{r['source_val'][d]['macro_f1']:5.1f}" for d in SOURCES)
        m, w, s = r["source_val"]["mean"], r["source_val"]["worst"], r["sketch"]
        sh = "-" if r["sharpness"] is None else f"{r['sharpness']:.4f}"
        print(f"{run:17s} {doms} {m['acc']:5.1f}/{m['macro_f1']:5.1f} {w['acc']:5.1f}/{w['macro_f1']:5.1f}"
              f" {s['acc']:5.1f}/{s['macro_f1']:5.1f} {r['sketch_acc_change']:+6.1f}"
              f" {r['source_separability']:5.1f} {sh:>7s}")

    t2 = json.loads((ROOT / "task2" / "results" / "final_eval.json").read_text())
    print(f"\nERM check: Task 2 source_only Sketch acc {t2['source_only']['target']['acc']:.2f},"
          f" Task 3 ERM {base['sketch']['acc']:.2f}")
    print("\nSketch acc/F1: DAN (Task 2, unlabeled Sketch) vs DAN-DG (Task 3, sources only)")
    for lam, r2, r3 in PAIRS:
        a, b = t2[r2]["target"], results[r3]["sketch"]
        print(f"  lambda {lam:>4}: DAN {a['acc']:5.1f}/{a['macro_f1']:5.1f}   DAN-DG {b['acc']:5.1f}/{b['macro_f1']:5.1f}")

    print("\nper-class Sketch accuracy (change vs ERM), dominant confusion")
    base_acc, base_cm = base["sketch_classes"]["acc"], base["sketch_classes"]["confusion"]
    for c, name in enumerate(CLASSES):
        wrong, n = dominant_confusion(base_cm, c)
        print(f"  erm {name:9s} {base_acc[name]:5.1f}  -> {wrong} ({n})")
    t2_dan = t2["dan"]["target_classes"]["acc"]
    for run in MAIN[1:] + STUDY:
        print(f" {run}")
        acc, cm = results[run]["sketch_classes"]["acc"], results[run]["sketch_classes"]["confusion"]
        for c, name in enumerate(CLASSES):
            wrong, n = dominant_confusion(cm, c)
            extra = f"   Task 2 DAN {t2_dan[name]:5.1f}" if run == "dan_dg" else ""
            print(f"  {name:9s} {acc[name]:5.1f} ({acc[name] - base_acc[name]:+5.1f})  -> {wrong} ({n}){extra}")

    out = RESULTS / "final_eval.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()