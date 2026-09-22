import json
from pathlib import Path

from common.seed import get_device
from shared.pacs_protocol import SOURCES
from task2.methods.source_only import SourceOnly as ERM
from task2.train import load_config
from task3.models.backbone import load_model
from task3.selection.source_validation import evaluate_sources, val_loaders

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["ERM"]


def main():
    cfg = load_config(ROOT / "task3" / "configs" / "erm.yaml")
    device = get_device()
    backbone, head, ckpt = load_model(ROOT / cfg["checkpoint"], device)
    assert ckpt["config"]["method"] == "source_only"

    val = evaluate_sources(backbone, head, val_loaders(), device)
    logged = json.loads((ROOT / "task2" / "results" / "source_only_train.json").read_text())["best_val"]
    diff = max(abs(val[d][k] - logged[d][k]) for d in SOURCES for k in ("acc", "macro_f1"))

    for d in SOURCES + ["mean"]:
        print(f"{d:13s} acc {val[d]['acc']:.2f}  macro-F1 {val[d]['macro_f1']:.2f}")
    w = val["worst"]
    print(f"worst         acc {w['acc']:.2f} ({w['acc_domain']})  macro-F1 {w['macro_f1']:.2f} ({w['macro_f1_domain']})")
    print(f"max difference from Task 2 log: {diff:.2e}")

    out = ROOT / "task3" / "results" / "erm_source_val.json"
    out.write_text(json.dumps({"name": "erm", "checkpoint": cfg["checkpoint"],
                               "epoch": ckpt["epoch"], "val": val}, indent=1))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()