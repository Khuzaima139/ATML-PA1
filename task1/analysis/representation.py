import umap


def fit_umap(features, seed, n_neighbors=15, min_dist=0.1, metric="cosine"):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(features)