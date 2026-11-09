"""Step B: distribution prediction with HistGradientBoostingClassifier, metric-aware decision.

Remaining time is cut into classes (CLASS_EDGES). The classifier (unweighted, so it estimates
P(class | features)) gives probabilities; the expected metric loss of predicting v is
sum_c P_c * omega_c * |v - rep_c|, with omega_c = mean W-bMAE window weight of training windows
in class c (window_weights: w_bin / (n_bin(rec) * sum_w(rec) * N_rec)) and rep_c = median training
target in class c. Its minimiser is the weighted median of rep_c with weights P_c * omega_c
("discrete"). "continuous" spreads each class uniformly over its interval and takes the point where
the weighted mass reaches 1/2. Evaluated with the dual CV (dual_cv.py), plain and calibrated.

Usage: python scripts/dist_hgb.py [B|B2]   (B2 = 30 s slices of remaining time + ">= 600")
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from dual_cv import OUT, STEP, dual_report, fmt, load_trainval, lofo_select, seen_splits, unseen_splits
from cv_utils import apply_bin_edges, fit_bin_edges
from stream_utils import summarize, window_weights

VARIANTS = {
    # B: 11 classes of growing width
    "B": np.array([0, 10, 20, 30, 40, 60, 90, 150, 225, 300, 450, 600.0]),
    # B2: 30 s slices of remaining time (k = 1..20) + a point class at the 600 s cap (">= 600").
    # Slices are in remaining time, not clock time, so nothing assumes the onsets lie on a 30 s grid.
    "B2": np.append(np.arange(0, 601, 30.0), 600.0),
}
CLASS_EDGES = VARIANTS["B"]
PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500, l2_regularization=5.0)
MAX_ITER = 400


def to_class(t, edges=CLASS_EDGES):
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)


def class_stats(dtr, edges=CLASS_EDGES):
    c = to_class(dtr.target.to_numpy(), edges)
    w = window_weights(dtr)
    n_cls = len(edges) - 1
    reps = np.array([np.median(dtr.target.to_numpy()[c == k]) for k in range(n_cls)])
    omega = np.array([w[c == k].mean() for k in range(n_cls)])
    return reps, omega / omega.sum()


def decide(P, reps, omega, continuous=False, edges=CLASS_EDGES):
    W = P * omega
    cum = np.cumsum(W, axis=1) / W.sum(axis=1, keepdims=True)
    k = (cum < 0.5).sum(axis=1)  # first class where the cumulative mass reaches 1/2
    k = np.minimum(k, P.shape[1] - 1)
    if not continuous:
        return reps[k]
    rows = np.arange(len(P))
    before = np.where(k > 0, cum[rows, np.maximum(k - 1, 0)], 0.0)
    mass = cum[rows, k] - before
    frac = np.clip((0.5 - before) / np.maximum(mass, 1e-12), 0, 1)
    lo, hi = edges[k], edges[k + 1]
    return lo + frac * (hi - lo)


DECISIONS = {"dist_discrete": False, "dist_continuous": True}


def run_clf(X, d, splits, edges=CLASS_EDGES, params=None, max_iter=MAX_ITER, keep_proba=False):
    """One classifier per fold; both decision rules scored on the same staged probabilities.
    keep_proba: also return, per decision rule, the held-out probability matrices at the selected
    number of trees (result[k] gets a 4th element: list of (n_te, n_classes) float32)."""
    params = PARAMS if params is None else params
    curves = {k: [] for k in DECISIONS}
    preds = {k: [] for k in DECISIONS}
    probas = []
    for tr, te in splits:
        dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
        reps, omega = class_stats(dtr, edges)
        bins = fit_bin_edges(X[tr])
        clf = HistGradientBoostingClassifier(max_iter=max_iter, max_features=0.5, early_stopping=False,
                                             random_state=0, **params)
        clf.fit(apply_bin_edges(X[tr], bins), to_class(dtr.target.to_numpy(), edges))
        assert len(clf.classes_) == len(reps)
        c = {k: [] for k in DECISIONS}
        p = {k: [] for k in DECISIONS}
        pr = []
        for n, P in enumerate(clf.staged_predict_proba(apply_bin_edges(X[te], bins)), start=1):
            if n % STEP == 0:
                for k, cont in DECISIONS.items():
                    q = decide(P, reps, omega, cont, edges)
                    c[k].append(summarize(dte, q)["wbmae"])
                    p[k].append(q.astype(np.float32))
                if keep_proba:
                    pr.append(P.astype(np.float32))
        for k in DECISIONS:
            curves[k].append(np.array(c[k]))
            preds[k].append(p[k])
        probas.append(pr)
    out = {k: lofo_select(curves[k], preds[k]) for k in DECISIONS}
    if keep_proba:  # same LOFO step choice as the decisions (identical curves)
        out = {k: v + (lofo_select(curves[k], probas)[0],) for k, v in out.items()}
    return out


if __name__ == "__main__":
    import sys
    variant = sys.argv[1] if len(sys.argv) > 1 else "B"
    edges = VARIANTS[variant]
    prefix = "" if variant == "B" else f"{variant}_"
    d, X, names = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    t0 = time.perf_counter()
    runs = {proto: run_clf(X, d, splits, edges) for proto, splits in protocols.items()}
    summary = json.loads((OUT / "summary.json").read_text())
    for name0 in DECISIONS:
        name = prefix + name0
        preds = {proto: runs[proto][name0][0] for proto in protocols}
        info = {f"n_iter_{proto}": runs[proto][name0][1] for proto in protocols}
        for proto in protocols:
            np.savez(OUT / f"{name}_{proto}_preds.npz", *preds[proto])
        res = dual_report(name, d, protocols, preds, dict(params=PARAMS, **info,
                                                           time_s=time.perf_counter() - t0))
        summary[name] = res
        print(fmt(res), f"  (n_iter {info})", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
