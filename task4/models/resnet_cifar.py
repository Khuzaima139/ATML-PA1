import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        net.maxpool = nn.Identity()
        self.net = net

    def pre(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        return n.layer2(n.layer1(x))

    def post(self, h):
        n = self.net
        h = n.layer4(n.layer3(h))
        return torch.flatten(n.avgpool(h), 1)

    def features(self, x):
        return self.post(self.pre(x))

    def forward(self, x):
        return self.net.fc(self.features(x))


if __name__ == "__main__":
    model = ResNetCIFAR()
    x = torch.randn(2, 3, 32, 32)
    print("pre:", tuple(model.pre(x).shape))
    print("features:", tuple(model.features(x).shape))
    print("logits:", tuple(model(x).shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))