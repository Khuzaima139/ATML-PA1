import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Identity()
        self.out_dim = 512

    def forward(self, x):
        return self.net(x)


def set_train_mode(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


if __name__ == "__main__":
    import torch

    from common.seed import get_device, set_seed

    set_seed(6304)
    device = get_device()
    model = Backbone().to(device)
    set_train_mode(model)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    before = bns[0].running_mean.clone()
    f = model(torch.randn(4, 3, 224, 224, device=device))
    print("device", device, "| feature", tuple(f.shape))
    print("model.training", model.training, "| all BN in eval", all(not m.training for m in bns), f"({len(bns)} BN layers)")
    print("running stats unchanged", torch.equal(before, bns[0].running_mean))
    print("gamma/beta trainable", all(m.weight.requires_grad and m.bias.requires_grad for m in bns))
    print("trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad))