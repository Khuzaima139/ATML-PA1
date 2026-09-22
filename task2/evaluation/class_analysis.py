import numpy as np
from sklearn.metrics import confusion_matrix

from shared.pacs_protocol import CLASSES


def per_class(y, pred):
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    acc = 100 * cm.diagonal() / cm.sum(1)
    return {"acc": dict(zip(CLASSES, acc.round(2).tolist())), "confusion": cm.tolist()}


def dominant_confusion(cm, c):
    row = np.array(cm[c])
    row[c] = 0
    j = int(row.argmax())
    return CLASSES[j], int(row[j])