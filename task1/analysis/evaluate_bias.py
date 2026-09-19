import torch
import torch.nn as nn

from common.metrics import acc_f1

BACKBONES = ["resnet50", "vit_b16", "clip_b32"]


def load_heads(ckpt_dir, feats):
    heads = {}
    for name in BACKBONES:
        head = nn.Linear(feats[name].shape[1], 10)
        head.load_state_dict(torch.load(ckpt_dir / f"{name}_head.pt"))
        head.eval()
        heads[name] = head
    return heads


@torch.no_grad()
def get_probs(heads, text_feats, logit_scale, feats):
    probs = {}
    for name in BACKBONES:
        probs[f"{name}_head"] = heads[name](feats[name]).softmax(dim=1)
    probs["clip_zeroshot"] = (logit_scale * feats["clip_b32"] @ text_feats.T).softmax(dim=1)
    return probs


def summarize(probs, labels):
    preds = probs.argmax(dim=1)
    metrics = acc_f1(labels.numpy(), preds.numpy())
    metrics["mean_max_conf"] = 100 * probs.max(dim=1).values.mean().item()
    return metrics


def consistency(preds_a, preds_b):
    return 100 * (preds_a == preds_b).float().mean().item()