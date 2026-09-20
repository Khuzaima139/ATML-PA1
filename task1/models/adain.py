# Architecture and pretrained weights from github.com/naoto0804/pytorch-AdaIN
# (Huang and Belongie, 2017, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization")
import torch
import torch.nn as nn

from task1.data.stl10 import ROOT, cfg


def conv(c_in, c_out):
    return [nn.ReflectionPad2d(1), nn.Conv2d(c_in, c_out, 3), nn.ReLU()]


def build_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 3, 1),
        *conv(3, 64), *conv(64, 64),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        *conv(64, 128), *conv(128, 128),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        *conv(128, 256), *conv(256, 256), *conv(256, 256), *conv(256, 256),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        *conv(256, 512),
    )


def build_decoder():
    return nn.Sequential(
        *conv(512, 256),
        nn.Upsample(scale_factor=2, mode="nearest"),
        *conv(256, 256), *conv(256, 256), *conv(256, 256), *conv(256, 128),
        nn.Upsample(scale_factor=2, mode="nearest"),
        *conv(128, 128), *conv(128, 64),
        nn.Upsample(scale_factor=2, mode="nearest"),
        *conv(64, 64),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
    )


def load_adain(device):
    encoder = build_encoder()
    state = torch.load(ROOT / cfg["cue_conflict"]["adain_vgg"])
    state = {k: v for k, v in state.items() if int(k.split(".")[0]) < len(encoder)}
    encoder.load_state_dict(state)

    decoder = build_decoder()
    decoder.load_state_dict(torch.load(ROOT / cfg["cue_conflict"]["adain_decoder"]))

    for net in (encoder, decoder):
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad = False
    return encoder, decoder


def mean_std(feat, eps=1e-5):
    n, c = feat.shape[:2]
    flat = feat.view(n, c, -1)
    mean = flat.mean(dim=2).view(n, c, 1, 1)
    std = (flat.var(dim=2) + eps).sqrt().view(n, c, 1, 1)
    return mean, std


def adain(content_feat, style_feat):
    c_mean, c_std = mean_std(content_feat)
    s_mean, s_std = mean_std(style_feat)
    return (content_feat - c_mean) / c_std * s_std + s_mean


@torch.no_grad()
def style_transfer(encoder, decoder, content, style, alpha):
    c = encoder(content)
    s = encoder(style)
    t = alpha * adain(c, s) + (1 - alpha) * c
    return decoder(t).clamp(0, 1)