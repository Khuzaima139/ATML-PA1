from pathlib import Path

import torch
import yaml

from shared.pacs_protocol import CLASSES
from task2.models.backbone import Backbone, set_train_mode
from task2.models.classifier_head import ClassifierHead

ROOT = Path(__file__).resolve().parents[2]
MAIN = ["erm", "dan_dg", "sam"]
STUDY = ["dan_dg_lambda0.1", "dan_dg_lambda10"]

__all__ = ["Backbone", "set_train_mode", "load_model", "load_run", "MAIN", "STUDY", "ROOT"]


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    backbone = Backbone().to(device)
    head = ClassifierHead(backbone.out_dim, len(CLASSES)).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    print(f"loaded {path.name} (epoch {ckpt['epoch']})")
    return backbone, head, ckpt


def load_run(run, device):
    if run == "erm":
        path = ROOT / yaml.safe_load((ROOT / "task3" / "configs" / "erm.yaml").read_text())["checkpoint"]
    else:
        path = ROOT / "checkpoints" / "task3" / f"{run}.pt"
    return load_model(path, device)