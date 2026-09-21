import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_rbf_mmd2(x, y, multipliers=(0.5, 1.0, 2.0)):
    z = torch.cat([x, y])
    sq = (z * z).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2 * z @ z.T).clamp_min(0)
    i, j = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    median = d2[i, j].median().detach()
    k = sum(torch.exp(-d2 / (m * median)) for m in multipliers)
    n = len(x)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


class DAN(nn.Module):
    uses_target = True

    def __init__(self, cfg, feat_dim, n_classes):
        super().__init__()
        self.lam = cfg["lambda_mmd"]

    def forward(self, f_s, logits_s, y_s, f_t, logits_t, progress):
        cls = F.cross_entropy(logits_s, y_s)
        mmd = multi_rbf_mmd2(f_s, f_t)
        return cls + self.lam * mmd, {"cls": cls.item(), "mmd": mmd.item()}