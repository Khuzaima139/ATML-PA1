from collections import Counter

import numpy as np


def accepted_by_class(u, tau, pred, fine, c100, c10):
    out = {}
    for c in np.unique(fine):
        mask = fine == c
        acc = mask & (u <= tau)
        absorbed = Counter(c10[p] for p in pred[acc])
        out[c100[c]] = {"accepted": int(acc.sum()), "of": int(mask.sum()), "predicted": dict(absorbed.most_common())}
    return out


def most_confident_accepted(u, tau, pred, fine, c100, c10, group):
    cases = []
    for c in np.unique(fine):
        idx = np.where((fine == c) & (u <= tau))[0]
        if len(idx) == 0:
            continue
        i = idx[np.argmin(u[idx])]
        cases.append({"group": group, "position": int(i), "unknown_class": c100[c],
                      "predicted": c10[pred[i]], "score": float(u[i]), "threshold": float(tau)})
    return cases