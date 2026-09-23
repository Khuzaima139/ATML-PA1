import numpy as np


def threshold(u_val, percentile=95):
    return float(np.percentile(u_val, percentile))


def accept_rate(u, tau):
    return 100 * float((u <= tau).mean())