import torch.nn.functional as F


class Vanilla:
    randaugment = False

    def loss(self, model, x, y):
        return F.cross_entropy(model(x), y)