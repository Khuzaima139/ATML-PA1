import torch.nn as nn
import torch.nn.functional as F


class SourceOnly(nn.Module):
    uses_target = False

    def __init__(self, cfg, feat_dim, n_classes):
        super().__init__()

    def forward(self, f_s, logits_s, y_s, f_t, logits_t, progress):
        cls = F.cross_entropy(logits_s, y_s)
        return cls, {"cls": cls.item()}