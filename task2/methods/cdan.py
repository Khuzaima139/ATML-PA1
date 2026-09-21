import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.models.domain_discriminator import DomainDiscriminator, grl_alpha


class CDAN(nn.Module):
    uses_target = True

    def __init__(self, cfg, feat_dim, n_classes):
        super().__init__()
        self.disc = DomainDiscriminator(feat_dim * n_classes)

    def forward(self, f_s, logits_s, y_s, f_t, logits_t, progress):
        cls = F.cross_entropy(logits_s, y_s)
        alpha = grl_alpha(progress)
        f = torch.cat([f_s, f_t])
        p = torch.softmax(torch.cat([logits_s, logits_t]), dim=1)
        g = (f.unsqueeze(2) * p.unsqueeze(1)).flatten(1)
        d = torch.cat([torch.zeros(len(f_s), dtype=torch.long),
                       torch.ones(len(f_t), dtype=torch.long)]).to(f.device)
        d_logits = self.disc(g, alpha)
        dom = F.cross_entropy(d_logits, d)
        d_acc = (d_logits.argmax(1) == d).float().mean().item() * 100
        return cls + dom, {"cls": cls.item(), "domain": dom.item(), "domain_acc": d_acc, "alpha": alpha}