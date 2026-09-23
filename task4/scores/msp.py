import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def msp(logits):
    return 1 - softmax(logits.astype(np.float64)).max(1)