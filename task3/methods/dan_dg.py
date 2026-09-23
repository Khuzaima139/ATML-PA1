from itertools import combinations

import torch.nn as nn
import torch.nn.functional as F

from task2.methods.dan import multi_rbf_mmd2


class DANDG(nn.Module):
    uses_target = False

    def __init__(self, cfg, feat_dim, n_classes):
        super().__init__()
        self.lam = cfg["lambda_dg"]
        self.n = cfg["per_source_batch"]

    def forward(self, f_s, logits_s, y_s, f_t, logits_t, progress):
        cls = F.cross_entropy(logits_s, y_s)
        pairs = list(combinations(f_s.split(self.n), 2))
        mmd = sum(multi_rbf_mmd2(a, b) for a, b in pairs) / len(pairs)
        return cls + self.lam * mmd, {"cls": cls.item(), "mmd": mmd.item()}