import json

import numpy as np
import torch
import torch.nn.functional as F

from common.seed import get_device
from shared.pacs import source_dataset
from shared.pacs_protocol import SEED, SOURCES
from task3.models.backbone import MAIN, ROOT, load_run

RADIUS = 0.05


def fixed_batch():
    rng = np.random.default_rng(SEED)
    xs, ys, ids = [], [], {}
    for d in SOURCES:
        ds = source_dataset(d, "val", train=False)
        idx = sorted(rng.choice(len(ds), size=32, replace=False).tolist())
        ids[d] = [ds.files[i] for i in idx]
        for i in idx:
            x, y = ds[i]
            xs.append(x)
            ys.append(y)
    return torch.stack(xs), torch.tensor(ys), ids


def sharpness(backbone, head, x, y):
    backbone.eval()
    head.eval()
    params = [p for m in (backbone, head) for p in m.parameters()]
    loss = F.cross_entropy(head(backbone(x)), y)
    grads = torch.autograd.grad(loss, params)
    norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(RADIUS * g / (norm + 1e-12))
        loss_adv = F.cross_entropy(head(backbone(x)), y)
        for p, s in zip(params, saved):
            p.copy_(s)
    return loss.item(), loss_adv.item()


def main():
    device = get_device()
    x, y, ids = fixed_batch()
    x, y = x.to(device), y.to(device)
    results = {"batch": ids}
    for run in MAIN:
        backbone, head, ckpt = load_run(run, device)
        loss, loss_adv = sharpness(backbone, head, x, y)
        results[run] = {"epoch": ckpt["epoch"], "loss": loss, "loss_perturbed": loss_adv,
                        "delta": loss_adv - loss}
        print(f"{run:7s} val loss {loss:.4f}  perturbed {loss_adv:.4f}  delta_sharp {loss_adv - loss:.4f}")
    out = ROOT / "task3" / "results" / "sharpness.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()