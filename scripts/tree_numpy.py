"""Pure-numpy inference for fitted sklearn HistGradientBoosting models (copied verbatim into submission.py).

``export_hgb(model)`` (build time) flattens every tree into padded arrays; ``hgb_raw`` /
``hgb_predict`` / ``hgb_predict_proba`` (inference) walk all trees at once with vectorised numpy,
so a 1-row call costs a few array ops per tree level instead of sklearn's per-call overhead.
Assumes numerical splits only and no missing values in the inputs (our inputs are pre-binned codes).
``pad_bin_edges`` / ``apply_padded_edges`` replace the per-feature searchsorted loop.
"""
import numpy as np


def export_hgb(model):
    """Fitted HistGradientBoosting{Regressor,Classifier} -> dict of numpy arrays."""
    preds = [p for it in model._predictors for p in it]  # iteration-major, class-minor
    n_k = len(model._predictors[0])
    M = max(p.nodes.shape[0] for p in preds)
    T = len(preds)
    feat = np.zeros((T, M), np.int32)
    thr = np.zeros((T, M), np.float64)
    left = np.zeros((T, M), np.int32)
    right = np.zeros((T, M), np.int32)
    leaf = np.ones((T, M), bool)
    value = np.zeros((T, M), np.float64)
    for t, p in enumerate(preds):
        n = p.nodes
        assert not n["is_categorical"].any()
        k = n.shape[0]
        feat[t, :k], thr[t, :k] = n["feature_idx"], n["num_threshold"]
        left[t, :k], right[t, :k] = n["left"], n["right"]
        leaf[t, :k], value[t, :k] = n["is_leaf"].astype(bool), n["value"]
    depth = int(max(p.nodes["depth"].max() for p in preds))
    return dict(feat=feat, thr=thr, left=left, right=right, leaf=leaf, value=value, depth=depth,
                n_classes=n_k, baseline=np.asarray(model._baseline_prediction, np.float64).reshape(-1))


def hgb_raw(trees, X):
    """X: (B, F) -> raw scores (B, n_classes), = sklearn's _raw_predict."""
    X = np.asarray(X, dtype=np.float64)
    T = trees["feat"].shape[0]
    B = X.shape[0]
    t = np.arange(T)[:, None]
    b = np.arange(B)[None, :]
    node = np.zeros((T, B), np.int64)
    for _ in range(trees["depth"]):
        go_left = X[b, trees["feat"][t, node]] <= trees["thr"][t, node]
        child = np.where(go_left, trees["left"][t, node], trees["right"][t, node])
        node = np.where(trees["leaf"][t, node], node, child)
    v = trees["value"][t, node]  # (T, B)
    K = trees["n_classes"]
    return v.reshape(T // K, K, B).sum(axis=0).T + trees["baseline"][None, :]


def hgb_predict(trees, X):
    """Regressor output (identity link)."""
    return hgb_raw(trees, X)[:, 0]


def hgb_predict_proba(trees, X):
    """Multiclass classifier probabilities (softmax of raw scores)."""
    r = hgb_raw(trees, X)
    e = np.exp(r - r.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def pad_bin_edges(edges):
    """List of per-feature sorted edge arrays -> (F, E) array padded with +inf, plus lengths."""
    E = max(len(e) for e in edges)
    out = np.full((len(edges), E), np.inf)
    for j, e in enumerate(edges):
        out[j, :len(e)] = e
    return out, np.array([len(e) for e in edges])


def apply_padded_edges(X, padded, lengths):
    """= np.searchsorted(e_j, X[:, j], side="right") for every feature j (NaN -> len(e_j))."""
    X = np.asarray(X, dtype=np.float64)
    codes = (X[:, :, None] >= padded[None, :, :]).sum(axis=-1)
    return np.where(np.isnan(X), lengths[None, :], codes).astype(np.float32)
