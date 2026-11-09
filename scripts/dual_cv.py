"""Dual cross-validation on train+val (test stays held out): unseen-subject CV + seen-subject CV.

- unseen: subject-grouped 5-fold (cv_utils.make_folds) -> new people.
- seen:   5 folds; in each, every subject with >= 2 nights has exactly one night held out (a
          different one per fold where possible), its other nights stay in training -> new nights
          of known people. Single-night subjects always train.
Decision score = (unseen mean + seen mean) / 2 (rule in CLAUDE.md).

Number of trees per fold = argmin of the mean held-out curve of the OTHER folds of the same
protocol. Calibration of fold k is learned from the other folds' held-out predictions, minus any
recording that is held out in fold k (seen folds can share recordings).

Usage: python scripts/dual_cv.py   (re-measures v1 and c5; writes ~/neuralbench/save/cv/dual/)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from calibrate_oof import fit_monotone_map
from cv_utils import N_FOLDS, apply_bin_edges, causal_smooth, fit_bin_edges, make_folds
from stream_utils import RANGE_NAMES, summarize, window_weights
from train_hgb import build_table, norm_weights

OUT = Path.home() / "neuralbench" / "save" / "cv" / "dual"
STEP = 25
HGB_CONFIGS = {  # name: (params, max trees)
    "v1": (dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0), 400),
    "c5": (dict(learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=10.0), 1000),
}


def unseen_splits(d):
    fold = make_folds(d)
    return [(fold != k, fold == k) for k in range(N_FOLDS)]


def seen_splits(d, n_folds=N_FOLDS, seed=0):
    rng = np.random.default_rng(seed)
    held = np.full((n_folds, len(d)), False)
    tl = d.timeline.to_numpy()
    for _, recs in d.groupby("subject").timeline.unique().items():
        if len(recs) < 2:
            continue
        recs = rng.permutation(recs)
        for k in range(n_folds):
            held[k] |= tl == recs[k % len(recs)]
    return [(~held[k], held[k]) for k in range(n_folds)]


def load_trainval():
    df, X, names = build_table()
    m = (df.split != "test").to_numpy()
    return df[m].reset_index(drop=True), X[m], names


def lofo_select(curves, preds):
    """Per fold, the step that minimises the mean curve of the other folds."""
    L = min(len(c) for c in curves)
    C = np.stack([c[:L] for c in curves])
    out, n_iter = [], []
    for k in range(len(curves)):
        j = int(np.delete(C, k, axis=0).mean(axis=0).argmin())
        out.append(preds[k][j])
        n_iter.append(STEP * (j + 1))
    return out, n_iter, int(STEP * (C.mean(axis=0).argmin() + 1))


def run_hgb(X, d, splits, params, max_iter):
    curves, preds = [], []
    for tr, te in splits:
        dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
        edges = fit_bin_edges(X[tr])
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=max_iter, max_features=0.5,
                                              early_stopping=False, random_state=0, **params)
        model.fit(apply_bin_edges(X[tr], edges), dtr.target, sample_weight=norm_weights(dtr))
        c, p = [], []
        for n, q in enumerate(model.staged_predict(apply_bin_edges(X[te], edges)), start=1):
            if n % STEP == 0:
                c.append(summarize(dte, q)["wbmae"])
                p.append(q.astype(np.float32))
        curves.append(np.array(c))
        preds.append(p)
    return lofo_select(curves, preds)


def calibrate(d, splits, preds, n_bins=20):
    out = []
    for k, (_, te) in enumerate(splits):
        held_k = set(d.timeline[te])
        ps, ds = [], []
        for j, (_, te_j) in enumerate(splits):
            if j == k:
                continue
            dj = d[te_j].reset_index(drop=True)
            keep = ~dj.timeline.isin(held_k).to_numpy()
            ps.append(preds[j][keep])
            ds.append(dj[keep])
        dcal = pd.concat(ds, ignore_index=True)
        xs, ys = fit_monotone_map(np.concatenate(ps), dcal.target.to_numpy(), window_weights(dcal), n_bins)
        out.append(np.interp(preds[k], xs, ys))
    return out


def score(d, splits, preds, smooth_a=0.0):
    sc = []
    for (_, te), p in zip(splits, preds):
        dte = d[te].reset_index(drop=True)
        sc.append(summarize(dte, causal_smooth(dte, p, smooth_a)))
    w = np.array([s["wbmae"] for s in sc])
    return dict(mean=float(w.mean()), std=float(w.std(ddof=1)), folds=w.round(3).tolist(),
                **{r: float(np.mean([s[r] for s in sc])) for r in RANGE_NAMES})


def dual_report(name, d, protocols, preds_by_protocol, extra=None):
    """preds_by_protocol: {protocol: list of per-fold held-out predictions}. Adds calibrated variant."""
    res = {"name": name, **(extra or {})}
    for variant in ("plain", "calib"):
        r = {}
        for proto, splits in protocols.items():
            p = preds_by_protocol[proto]
            if variant == "calib":
                p = calibrate(d, splits, p)
            r[proto] = score(d, splits, p)
        r["dual"] = (r["unseen"]["mean"] + r["seen"]["mean"]) / 2
        res[variant] = r
    return res


def fmt(res):
    lines = []
    for v in ("plain", "calib"):
        r = res[v]
        lines.append(f"  {res['name']:10s} {v:5s}  unseen {r['unseen']['mean']:.2f} ± {r['unseen']['std']:.2f}  "
                     f"seen {r['seen']['mean']:.2f} ± {r['seen']['std']:.2f}  DUAL {r['dual']:.2f}  | "
                     + "  ".join(f"{p} [0,40) {r[p]['[0,40)']:.1f}" for p in ("unseen", "seen")))
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d, X, names = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    for k, (tr, te) in enumerate(protocols["seen"]):
        dte = d[te]
        assert set(dte.subject).issubset(set(d[tr].subject)), "seen fold has a subject without train nights"
    print("seen folds: held-out recordings per fold",
          [int(d[te].timeline.nunique()) for _, te in protocols["seen"]], flush=True)
    summary = {}
    for name, (params, max_iter) in HGB_CONFIGS.items():
        t0 = time.perf_counter()
        preds, info = {}, {}
        for proto, splits in protocols.items():
            preds[proto], n_iter, n_mc = run_hgb(X, d, splits, params, max_iter)
            info[f"n_iter_{proto}"] = n_iter
            np.savez(OUT / f"{name}_{proto}_preds.npz", *preds[proto])
        res = dual_report(name, d, protocols, preds, dict(params=params, **info,
                                                           time_s=time.perf_counter() - t0))
        summary[name] = res
        print(fmt(res), f"  (n_iter {info}, {res['time_s']:.0f} s)", flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
