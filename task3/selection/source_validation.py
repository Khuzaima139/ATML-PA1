from torch.utils.data import DataLoader

from shared.pacs import source_dataset
from shared.pacs_protocol import SOURCES
from task2.evaluation.metrics import source_val_metrics
from task3.evaluation.domain_metrics import with_worst


def val_loaders():
    return {d: DataLoader(source_dataset(d, "val", train=False), batch_size=64) for d in SOURCES}


def evaluate_sources(backbone, head, loaders, device):
    return with_worst(source_val_metrics(backbone, head, loaders, device))


def selection_score(val):
    return val["mean"]["macro_f1"]