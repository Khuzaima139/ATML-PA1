import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import Normalize

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x):
        return F.normalize(self.clip_model.encode_image(x), dim=-1)


class FrozenBackbone(nn.Module):
    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.norm = Normalize(mean, std)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.net(self.norm(x))


def load_backbones(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()

    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    backbones = {
        "resnet50": FrozenBackbone(resnet, IMAGENET_MEAN, IMAGENET_STD),
        "vit_b16": FrozenBackbone(vit, IMAGENET_MEAN, IMAGENET_STD),
        "clip_b32": FrozenBackbone(CLIPImageEncoder(clip_model), CLIP_MEAN, CLIP_STD),
    }
    for b in backbones.values():
        b.to(device)
    return backbones, clip_model, tokenizer


@torch.no_grad()
def clip_text_features(clip_model, tokenizer, classes, device):
    prompts = [f"a photo of a {c}." for c in classes]
    tokens = tokenizer(prompts).to(device)
    return F.normalize(clip_model.encode_text(tokens), dim=-1)


def clip_zero_shot_logits(clip_model, image_feats, text_feats):
    return clip_model.logit_scale.exp() * image_feats @ text_feats.T

def clip_logit_scale():
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    return clip_model.logit_scale.exp().item()

if __name__ == "__main__":
    from common.seed import get_device

    device = get_device()
    backbones, clip_model, tokenizer = load_backbones(device)
    x = torch.rand(2, 3, 224, 224, device=device)

    for name, b in backbones.items():
        f = b(x)
        print(name, tuple(f.shape))

    classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    text = clip_text_features(clip_model, tokenizer, classes, device)
    img = backbones["clip_b32"](x)
    print("clip text", tuple(text.shape))
    print("clip image norms", img.norm(dim=-1).tolist())
    print("zero-shot logits", tuple(clip_zero_shot_logits(clip_model, img, text).shape))