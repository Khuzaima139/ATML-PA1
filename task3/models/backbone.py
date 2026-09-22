import torch

from shared.pacs_protocol import CLASSES
from task2.models.backbone import Backbone, set_train_mode
from task2.models.classifier_head import ClassifierHead

__all__ = ["Backbone", "set_train_mode", "load_model"]


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    backbone = Backbone().to(device)
    head = ClassifierHead(backbone.out_dim, len(CLASSES)).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    print(f"loaded {path.name} (epoch {ckpt['epoch']})")
    return backbone, head, ckpt