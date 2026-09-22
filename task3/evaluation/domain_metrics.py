from shared.pacs_protocol import SOURCES


def with_worst(val):
    worst = {}
    for k in ("acc", "macro_f1"):
        d = min(SOURCES, key=lambda d: val[d][k])
        worst[k], worst[f"{k}_domain"] = val[d][k], d
    return {**val, "worst": worst}