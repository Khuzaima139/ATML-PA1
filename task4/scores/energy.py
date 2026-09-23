import numpy as np


def energy(logits):
    z = logits.astype(np.float64)
    m = z.max(1)
    return -(m + np.log(np.exp(z - m[:, None]).sum(1)))