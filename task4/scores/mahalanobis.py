import numpy as np


def fit(feat, labels, num_classes=10, eps=1e-6):
    feat = feat.astype(np.float64)
    means = np.stack([feat[labels == c].mean(0) for c in range(num_classes)])
    var = ((feat - means[labels]) ** 2).mean(0) + eps
    return means, var


def mahalanobis(feat, means, var):
    feat = feat.astype(np.float64)
    dist = np.stack([((feat - m) ** 2 / var).sum(1) for m in means], axis=1)
    return dist.min(1)