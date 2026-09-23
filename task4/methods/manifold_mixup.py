import torch


def mix_different_classes(h, y, lam):
    perm = torch.randperm(len(y), device=y.device)
    keep = y != y[perm]
    return lam * h[keep] + (1 - lam) * h[perm][keep], keep.float().mean().item()