import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(u_known, u_unknown):
    y = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]
    return 100 * roc_auc_score(y, np.r_[u_known, u_unknown])