"""Monotone calibration of HGB out-of-fold predictions, leak-free across the 5 CV folds.

For each fold k the map pred -> corrected pred is learned only from the OOF rows of the other
folds: predictions are cut into bins of equal W-bMAE weight, each bin gives (weighted median
pred, weighted median target) -- medians because the metric is L1 -- and the bin targets are made
non-decreasing with weighted isotonic regression; np.interp between knots, flat outside.
Variants: calibration, smoothing (a=0.3), calibration->smoothing, smoothing->calibration.

Usage: python scripts/calibrate_oof.py [oof.parquet]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cv_utils import N_FOLDS, causal_smooth
from stream_utils import RANGE_NAMES, summarize, weighted_median, window_weights

SAVE = Path.home() / "neuralbench" / "save" / "cv"
A = 0.3


def fit_monotone_map(pred, target, w, n_bins=20):
    order = np.argsort(pred)
    p, t, w = pred[order], target[order], w[order]
    cum = np.cumsum(w) / w.sum()
    edges = np.searchsorted(cum, np.linspace(0, 1, n_bins + 1)[1:-1])
    xs, ys, ws = [], [], []
    for idx in np.split(np.arange(len(p)), edges):
        if len(idx) == 0:
            continue
        xs.append(weighted_median(p[idx], w[idx]))
        ys.append(weighted_median(t[idx], w[idx]))
        ws.append(w[idx].sum())
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
    ys = IsotonicRegression(increasing=True).fit_transform(xs, ys, sample_weight=ws)
    xs, keep = np.unique(xs, return_index=True)  # np.interp needs increasing x
    return xs, ys[keep]


def cv_scores(d, fold, p):
    return [summarize(d[fold == k], p[fold == k]) for k in range(N_FOLDS)]


def calibrated(d, fold, p, n_bins):
    out = np.empty_like(p)
    for k in range(N_FOLDS):
        tr = fold != k
        dk = d[tr]
        xs, ys = fit_monotone_map(p[tr], dk.target.to_numpy(), window_weights(dk), n_bins)
        out[~tr] = np.interp(p[~tr], xs, ys)
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAVE / "hgb_v1_oof.parquet"
    d = pd.read_parquet(path).reset_index(drop=True)
    fold, p = d.fold.to_numpy(), d.pred.to_numpy()
    p_s = causal_smooth(d, p, A)
    variants = {"none": p, f"smooth a={A}": p_s}
    for nb in (10, 20, 40):
        variants[f"calib{nb}"] = calibrated(d, fold, p, nb)
    variants["calib20 -> smooth"] = causal_smooth(d, variants["calib20"], A)
    variants["smooth -> calib20"] = calibrated(d, fold, p_s, 20)

    base = np.array([r["wbmae"] for r in cv_scores(d, fold, p)])
    out = {}
    print(f"{'variant':20s} {'CV W-bMAE':>15s} {'paired diff':>14s}  " + " ".join(f"{r:>9s}" for r in RANGE_NAMES))
    for name, q in variants.items():
        sc = cv_scores(d, fold, q)
        w = np.array([r["wbmae"] for r in sc])
        diff = w - base
        rng = {r: float(np.mean([s[r] for s in sc])) for r in RANGE_NAMES}
        out[name] = dict(mean=w.mean(), std=w.std(ddof=1), folds=w.tolist(),
                         diff_mean=diff.mean(), diff_std=diff.std(ddof=1), **rng)
        print(f"{name:20s} {w.mean():7.2f} ± {w.std(ddof=1):4.2f}  {diff.mean():+6.2f} ± {diff.std(ddof=1):4.2f}  "
              + " ".join(f"{rng[r]:9.1f}" for r in RANGE_NAMES))

    # map learned on all OOF rows (what a submission would use), for inspection
    xs, ys = fit_monotone_map(p, d.target.to_numpy(), window_weights(d), 20)
    print("\nfull-OOF map (pred -> calibrated):", ", ".join(f"{x:.0f}->{y:.0f}" for x, y in zip(xs, ys)))
    out["full_map"] = dict(x=xs.tolist(), y=ys.tolist())
    (path.parent / (path.stem + "_calibration.json")).write_text(json.dumps(out, indent=2, default=float))
