import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.models.domain_discriminator import DomainDiscriminator, grl_alpha


class DANNL2(nn.Module):
    uses_target = True

    def __init__(self, cfg, feat_dim, n_classes):
        super().__init__()
        self.disc = DomainDiscriminator(feat_dim)

    def forward(self, f_s, logits_s, y_s, f_t, logits_t, progress):
        cls = F.cross_entropy(logits_s, y_s)
        alpha = grl_alpha(progress)
        f = F.normalize(torch.cat([f_s, f_t]), dim=1)
        d = torch.cat([torch.zeros(len(f_s), dtype=torch.long),
                       torch.ones(len(f_t), dtype=torch.long)]).to(f.device)
        d_logits = self.disc(f, alpha)
        dom = F.cross_entropy(d_logits, d)
        d_acc = (d_logits.argmax(1) == d).float().mean().item() * 100
        return cls + dom, {"cls": cls.item(), "domain": dom.item(), "domain_acc": d_acc, "alpha": alpha}