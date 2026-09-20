def classify_predictions(preds, manifest, classes):
    shape_idx = [classes.index(m["shape"]) for m in manifest]
    texture_idx = [classes.index(m["texture"]) for m in manifest]

    labels = []
    for p, s, t in zip(preds.tolist(), shape_idx, texture_idx):
        if p == s:
            labels.append("shape")
        elif p == t:
            labels.append("texture")
        else:
            labels.append("other")
    return labels


def shape_bias_and_coverage(decision_labels):
    n_shape = decision_labels.count("shape")
    n_texture = decision_labels.count("texture")
    n_total = len(decision_labels)
    denom = n_shape + n_texture
    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_total - denom,
        "n_total": n_total,
        "shape_bias": 100 * n_shape / denom if denom else float("nan"),
        "coverage": 100 * denom / n_total,
    }