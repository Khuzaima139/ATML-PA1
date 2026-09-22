import json

import torch
from torch.utils.data import ConcatDataset, DataLoader

from common.seed import get_device
from shared.pacs import source_dataset, target_dataset
from shared.pacs_protocol import SOURCES
from task2.models.backbone import Backbone
from task2.train import ROOT

RUNS = ["imagenet", "source_only", "dan", "dan_lambda0.1", "dan_lambda10",
        "dann", "cdan", "dann_l2", "cdan_l2"]


@torch.no_grad()
def features(backbone, ds, device):
    out = []
    for batch in DataLoader(ds, batch_size=64):
        x = batch[0] if isinstance(batch, list) else batch
        out.append(backbone(x.to(device)).cpu())
    return torch.cat(out)


def main():
    device = get_device()
    src = ConcatDataset([source_dataset(d, "val", train=False) for d in SOURCES])
    tgt = target_dataset(train=False)
    results = {}
    print(f"{'run':15s} {'epoch':>5s} {'|f| src':>9s} {'|f| sketch':>11s} {'std src':>8s} {'std sketch':>10s}")
    for run in RUNS:
        backbone = Backbone().to(device)
        epoch = None
        if run != "imagenet":
            ckpt = torch.load(ROOT / "checkpoints" / "task2" / f"{run}.pt", map_location=device)
            backbone.load_state_dict(ckpt["backbone"])
            epoch = ckpt["epoch"]
        backbone.eval()
        fs, ft = features(backbone, src, device), features(backbone, tgt, device)
        results[run] = {"epoch": epoch,
                        "norm_src": fs.norm(dim=1).mean().item(), "norm_sketch": ft.norm(dim=1).mean().item(),
                        "std_src": fs.std(0).mean().item(), "std_sketch": ft.std(0).mean().item()}
        r = results[run]
        print(f"{run:15s} {str(epoch or '-'):>5s} {r['norm_src']:9.1f} {r['norm_sketch']:11.1f}"
              f" {r['std_src']:8.3f} {r['std_sketch']:10.3f}")
    out = ROOT / "task2" / "results" / "feature_norms.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()