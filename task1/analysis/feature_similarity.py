import torch.nn.functional as F


def cosine_stability(clean, transformed):
    return 100 * F.cosine_similarity(clean, transformed, dim=1).mean().item()