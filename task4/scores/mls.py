import numpy as np


def mls(logits):
    return -logits.astype(np.float64).max(1)