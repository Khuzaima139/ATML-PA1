import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset

from shared.pacs import source_dataset, target_dataset
from shared.pacs_protocol import SEED, SOURCES


@torch.no_grad()
def extract(backbone, ds, device):
    backbone.eval()
    out = []
    for batch in DataLoader(ds, batch_size=64):
        x = batch[0] if isinstance(batch, list) else batch
        out.append(backbone(x.to(device)).cpu())
    return torch.cat(out).numpy()


def separability_sets():
    src = ConcatDataset([source_dataset(d, "val", train=False) for d in SOURCES])
    tgt = target_dataset(train=False)
    idx = np.random.default_rng(SEED).choice(len(tgt), size=len(src), replace=False)
    return src, Subset(tgt, sorted(idx.tolist()))


def domain_separability(backbone, src, tgt, device):
    fs, ft = extract(backbone, src, device), extract(backbone, tgt, device)
    X = np.concatenate([fs, ft])
    d = np.r_[np.zeros(len(fs)), np.ones(len(ft))]
    X_tr, X_te, d_tr, d_te = train_test_split(X, d, test_size=0.3, stratify=d, random_state=SEED)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(X_tr, d_tr)
    return 100 * clf.score(X_te, d_te)