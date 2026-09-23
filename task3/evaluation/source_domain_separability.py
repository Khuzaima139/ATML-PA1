import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from common.seed import get_device
from shared.pacs import source_dataset
from shared.pacs_protocol import SEED, SOURCES
from task2.evaluation.domain_separability import extract
from task3.models.backbone import MAIN, ROOT, STUDY, load_run


def balanced_sets():
    sets = [source_dataset(d, "val", train=False) for d in SOURCES]
    n = min(len(s) for s in sets)
    rng = np.random.default_rng(SEED)
    return [Subset(s, sorted(rng.choice(len(s), size=n, replace=False).tolist())) for s in sets]


def source_separability(backbone, sets, device):
    feats = [extract(backbone, s, device) for s in sets]
    X = np.concatenate(feats)
    d = np.concatenate([np.full(len(f), i) for i, f in enumerate(feats)])
    X_tr, X_te, d_tr, d_te = train_test_split(X, d, test_size=0.3, stratify=d, random_state=SEED)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(X_tr, d_tr)
    return 100 * clf.score(X_te, d_te), X


def main():
    device = get_device()
    sets = balanced_sets()
    print(f"balanced source-val features: {len(sets[0])} per domain")
    results = {}
    for run in MAIN + STUDY:
        backbone, _, ckpt = load_run(run, device)
        sep, X = source_separability(backbone, sets, device)
        results[run] = {"epoch": ckpt["epoch"], "separability": sep,
                        "feature_norm": float(np.linalg.norm(X, axis=1).mean()),
                        "feature_std": float(X.std(0).mean())}
        r = results[run]
        print(f"{run:17s} separability {sep:5.1f}  |f| {r['feature_norm']:6.1f}  std {r['feature_std']:.3f}")
    out = ROOT / "task3" / "results" / "source_separability.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()