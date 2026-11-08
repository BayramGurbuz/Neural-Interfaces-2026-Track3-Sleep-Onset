"""Regularization grid for HGB, compared on the 5-fold subject-grouped CV (train+val).

Per config and fold, one fit with the full tree budget on pre-binned features (bin edges from
the fold's training rows, see cv_utils.fit_bin_edges); the held-out fold is scored every 25 trees. Number of trees for fold k = argmin of the MEAN curve of the other 4 folds
(leave-one-fold-out choice: fold k's own score never picks its n_iter). Reported per config:
plain, causal smoothing (a=0.3) and leak-free monotone calibration (20 bins), mean ± std over
folds. OOF predictions of every config are saved for later use.

Usage: python scripts/cv_hgb_grid.py   (writes ~/neuralbench/save/cv/grid/)
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from calibrate_oof import calibrated
from cv_utils import N_FOLDS, apply_bin_edges, causal_smooth, fit_bin_edges, make_folds
from stream_utils import RANGE_NAMES, summarize
from train_hgb import build_table, norm_weights

OUT = Path.home() / "neuralbench" / "save" / "cv" / "grid"
OUT.mkdir(parents=True, exist_ok=True)
STEP = 25
CONFIGS = {  # name: (learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization)
    "c0_v1": (0.05, 31, 200, 1.0),
    "c1": (0.05, 15, 500, 1.0),
    "c2": (0.05, 7, 1000, 5.0),
    "c3": (0.03, 15, 500, 5.0),
    "c4": (0.03, 7, 200, 10.0),
    "c5": (0.02, 15, 1000, 10.0),
    "c6": (0.02, 7, 500, 1.0),
    "c7": (0.02, 31, 1000, 10.0),
}
MAX_ITER = {0.05: 400, 0.03: 650, 0.02: 1000}


def run_fold(X, d, tr, te, lr, leaves, msl, l2):
    """One fit with the full tree budget; held-out curve via staged_predict every STEP trees."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
        l2_regularization=l2, max_features=0.5, early_stopping=False, random_state=0,
        max_iter=MAX_ITER[lr])
    dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
    edges = fit_bin_edges(X[tr])
    model.fit(apply_bin_edges(X[tr], edges), dtr.target, sample_weight=norm_weights(dtr))
    curve, preds = [], []
    for n, p in enumerate(model.staged_predict(apply_bin_edges(X[te], edges)), start=1):
        if n % STEP == 0:
            curve.append(summarize(dte, p)["wbmae"])
            preds.append(p.astype(np.float32))
    return np.array(curve), np.stack(preds)


def score_folds(d, fold, p):
    sc = [summarize(d[fold == k], p[fold == k]) for k in range(N_FOLDS)]
    w = np.array([s["wbmae"] for s in sc])
    return dict(mean=float(w.mean()), std=float(w.std(ddof=1)), folds=w.round(3).tolist(),
                **{r: float(np.mean([s[r] for s in sc])) for r in RANGE_NAMES})


if __name__ == "__main__":
    df, X, names = build_table()
    m_tv = (df.split != "test").to_numpy()
    d = df[m_tv].reset_index(drop=True)
    X = X[m_tv]
    fold = make_folds(d)
    summary = {}
    for cname, (lr, leaves, msl, l2) in CONFIGS.items():
        t0 = time.perf_counter()
        curves, preds = [], []
        for k in range(N_FOLDS):
            c, p = run_fold(X, d, fold != k, fold == k, lr, leaves, msl, l2)
            curves.append(c)
            preds.append(p)
            print(f"  {cname} fold {k}: best {c.min():.2f} @ {STEP * (c.argmin() + 1)} trees, "
                  f"ran {STEP * len(c)} ({time.perf_counter() - t0:.0f} s)", flush=True)
        L = min(len(c) for c in curves)
        C = np.stack([c[:L] for c in curves])
        oof = np.empty(len(d))
        n_iter = []
        for k in range(N_FOLDS):
            j = int(np.delete(C, k, axis=0).mean(axis=0).argmin())
            n_iter.append(STEP * (j + 1))
            oof[fold == k] = preds[k][j]
        res = dict(params=dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                               l2_regularization=l2),
                   n_iter_lofo=n_iter, n_iter_meancurve=int(STEP * (C.mean(axis=0).argmin() + 1)),
                   plain=score_folds(d, fold, oof),
                   smooth03=score_folds(d, fold, causal_smooth(d, oof, 0.3)),
                   calib20=score_folds(d, fold, calibrated(d, fold, oof, 20)),
                   time_s=time.perf_counter() - t0)
        summary[cname] = res
        d.assign(fold=fold, pred=oof)[["split", "timeline", "stop", "subject", "target", "fold",
                                       "pred"]].to_parquet(OUT / f"{cname}_oof.parquet")
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{cname}] lr={lr} leaves={leaves} msl={msl} l2={l2} n_iter={n_iter}  "
              f"plain {res['plain']['mean']:.2f}±{res['plain']['std']:.2f}  "
              f"smooth {res['smooth03']['mean']:.2f}  calib {res['calib20']['mean']:.2f}±{res['calib20']['std']:.2f}  "
              f"[0,40) {res['plain']['[0,40)']:.1f}/{res['calib20']['[0,40)']:.1f}  ({res['time_s']:.0f} s)",
              flush=True)
