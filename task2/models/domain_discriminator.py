import math

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grl_alpha(p):
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 2),
        )

    def forward(self, f, alpha):
        return self.net(GradReverse.apply(f, alpha))
    