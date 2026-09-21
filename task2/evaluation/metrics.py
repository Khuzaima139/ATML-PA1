import torch

from common.metrics import acc_f1


@torch.no_grad()
def predict(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    ys, preds = [], []
    for x, y in loader:
        preds.append(head(backbone(x.to(device))).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def source_val_metrics(backbone, head, loaders, device):
    out = {d: acc_f1(*predict(backbone, head, loader, device)) for d, loader in loaders.items()}
    out["mean"] = {k: sum(out[d][k] for d in loaders) / len(loaders) for k in ("acc", "macro_f1")}
    return out