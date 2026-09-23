import torch
import torch.nn as nn
import torch.nn.functional as F

from task4.methods.manifold_mixup import mix_different_classes
from task4.models.resnet_cifar import ResNetCIFAR

K = 10


class ProserNet(ResNetCIFAR):
    def __init__(self, num_dummy=5):
        super().__init__()
        self.dummy = nn.Linear(512, num_dummy)

    def open_logits(self, f):
        known = self.net.fc(f)
        dummy = self.dummy(f).max(dim=1, keepdim=True).values
        return torch.cat([known, dummy], dim=1)


class Proser:
    randaugment = False

    def __init__(self, cfg):
        self.beta = cfg["beta"]
        self.gamma = cfg["gamma"]
        self.mix = torch.distributions.Beta(cfg["mix_alpha"], cfg["mix_alpha"])
        self.logs = {}

    def loss(self, model, x, y):
        half = len(x) // 2
        xa, ya, xb, yb = x[:half], y[:half], x[half:], y[half:]

        out_a = model.open_logits(model.features(xa))
        l_cls = F.cross_entropy(out_a, ya)
        masked = out_a.scatter(1, ya[:, None], -1e9)
        l_dummy = F.cross_entropy(masked, torch.full_like(ya, K))
        l1 = l_cls + self.beta * l_dummy

        lam = self.mix.sample().item()
        h_mix, keep = mix_different_classes(model.pre(xb), yb, lam)
        if len(h_mix) > 0:
            out_b = model.open_logits(model.post(h_mix))
            l_mix = F.cross_entropy(out_b, torch.full((len(out_b),), K, device=y.device))
        else:
            l_mix = torch.zeros((), device=y.device)

        self.logs = {"l_cls": l_cls.item(), "l_dummy": l_dummy.item(), "l_mix": l_mix.item(), "mix_keep": keep}
        return l1 + self.gamma * l_mix