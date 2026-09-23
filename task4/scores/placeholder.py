import numpy as np

from task4.scores.msp import softmax


def placeholder(logits, dummy, temperature=1024.0):
    z = np.concatenate([logits, dummy.max(1, keepdims=True)], axis=1).astype(np.float64)
    p = softmax(z / temperature)
    return p[:, -1] - p[:, :-1].max(1)