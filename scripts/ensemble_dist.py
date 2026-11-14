"""Step B/C follow-up: distribution-level ensembles of dist_hgb classifiers, dual CV.

Members (same feature table, chosen with the first argument = an ablation_c group name or "base"):
  B      : class edges B,  dist_hgb.PARAMS
  B_h1   : class edges B,  deeper / less regularised
  B_h2   : class edges B,  shallower / more regularised
  B2     : class edges B2 (30 s slices + point class at 600), dist_hgb.PARAMS
Each member's held-out probabilities (at its own LOFO-selected number of trees) are cached.

Ensembles (probabilities averaged first, then one continuous weighted-median decision):
  (a) B + B2: different classes -> every distribution is spread uniformly inside its classes onto a
      common grid (union of both edge sets + the point class at 600); omega per grid cell = mean
      window weight of the fold's training windows in that cell. B and B2 alone are also scored on
      the same grid, as references for the decision change.
  (b) B + B_h1 + B_h2 (same classes): mean of P, decision with B's own omega.
Scored with dual_report (plain + leak-free calibration) and per-bin contributions.

Usage: python scripts/ensemble_dist.py [feature_set]   (default dyn)
"""
import json
import sys
import time

import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from dist_hgb import PARAMS, VARIANTS, class_stats, decide, run_clf, to_class
from dual_cv import OUT, dual_report, fmt, seen_splits, unseen_splits
from report_bins import contributions
from stream_utils import RANGE_NAMES, window_weights

MEMBERS = {  # name: (edges, params, max_iter)
    "B": (VARIANTS["B"], PARAMS, 400),
    "B_h1": (VARIANTS["B"], dict(learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=300,
                                 l2_regularization=2.0), 600),
    "B_h2": (VARIANTS["B"], dict(learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=1000,
                                 l2_regularization=10.0), 600),
    "B2": (VARIANTS["B2"], PARAMS, 400),
}
GRID = np.append(np.union1d(VARIANTS["B"], VARIANTS["B2"]), 600.0)  # last cell = point at 600


def to_grid(P, edges, grid=GRID):
    """(n, C) class probabilities -> (n, J) grid-cell probabilities, uniform inside each class."""
    A = np.zeros((len(edges) - 1, len(grid) - 1))
    for c in range(len(edges) - 1):
        lo, hi = edges[c], edges[c + 1]
        if hi == lo:  # point class
            A[c, len(grid) - 2] = 1.0
            continue
        for j in range(len(grid) - 1):
            g0, g1 = grid[j], grid[j + 1]
            if g1 > g0:
                A[c, j] = max(0.0, min(hi, g1) - max(lo, g0)) / (hi - lo)
    return P @ A


def grid_omega(dtr, grid=GRID):
    c = to_class(dtr.target.to_numpy(), grid)
    w = window_weights(dtr)
    om = np.array([w[c == k].mean() if (c == k).any() else 0.0 for k in range(len(grid) - 1)])
    return om / om.sum()


def member_probas(name, feat_set, X, d, protocols):
    path = OUT / f"ens_{feat_set}_{name}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["probas"].item(), z["n_iter"].item()
    edges, params, max_iter = MEMBERS[name]
    probas, n_iter = {}, {}
    for proto, splits in protocols.items():
        part = OUT / f"ens_{feat_set}_{name}_{proto}.part.npz"  # per-protocol checkpoint (survives a shutdown)
        if part.exists():
            z = np.load(part, allow_pickle=True)
            probas[proto] = [z[f"arr_{i}"] for i in range(len(splits))]
            n_iter[proto] = z["n_iter"].tolist()
            print(f"  member {name} {proto}: loaded from {part.name}", flush=True)
            continue
        t0 = time.perf_counter()
        r = run_clf(X, d, splits, edges, params, max_iter, keep_proba=True)["dist_continuous"]
        probas[proto], n_iter[proto] = r[3], r[1]
        np.savez(part, *r[3], n_iter=np.array(r[1]))
        print(f"  member {name} {proto}: n_iter {r[1]} ({time.perf_counter() - t0:.0f} s)", flush=True)
    np.savez(path, probas=np.array(probas, dtype=object), n_iter=np.array(n_iter, dtype=object))
    return probas, n_iter


def ensemble_preds(d, protocols, members, probas, mode):
    """mode 'own': same classes, B omega; 'grid': common grid."""
    out = {}
    for proto, splits in protocols.items():
        out[proto] = []
        for k, (tr, _) in enumerate(splits):
            dtr = d[tr].reset_index(drop=True)
            if mode == "own":
                edges = MEMBERS[members[0]][0]
                _, omega = class_stats(dtr, edges)
                P = np.mean([probas[m][proto][k] for m in members], axis=0)
            else:
                edges, omega = GRID, grid_omega(dtr)
                P = np.mean([to_grid(probas[m][proto][k], MEMBERS[m][0]) for m in members], axis=0)
            out[proto].append(decide(P.astype(np.float64), None, omega, True, edges).astype(np.float32))
    return out


ENSEMBLES = {  # name: (members, mode)
    "B": (["B"], "own"), "B_h1": (["B_h1"], "own"), "B_h2": (["B_h2"], "own"), "B2": (["B2"], "own"),
    "B@grid": (["B"], "grid"), "B2@grid": (["B2"], "grid"),
    "(a) B+B2@grid": (["B", "B2"], "grid"),
    "(b) B+B_h1+B_h2": (["B", "B_h1", "B_h2"], "own"),
    "(c) B+B_h1+B_h2+B2@grid": (["B", "B_h1", "B_h2", "B2"], "grid"),
}

if __name__ == "__main__":
    feat_set = sys.argv[1] if len(sys.argv) > 1 else "dyn"
    wg, sg, drop = ABLATIONS[feat_set] if feat_set != "base" else ((), (), False)
    df, X, names = build_table_ext(wg, sg, drop)
    m = (df.split != "test").to_numpy()
    d, X = df[m].reset_index(drop=True), X[m]
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    print(f"feature set {feat_set}: {X.shape[1]} features", flush=True)
    probas, n_iters = {}, {}
    for name in MEMBERS:
        probas[name], n_iters[name] = member_probas(name, feat_set, X, d, protocols)
    summary = {}
    for ens, (members, mode) in ENSEMBLES.items():
        preds = ensemble_preds(d, protocols, members, probas, mode)
        res = dual_report(ens, d, protocols, preds, dict(members=members, mode=mode,
                                                         n_iter={mm: n_iters[mm] for mm in members}))
        res["contrib_plain"] = {p: dict(zip(RANGE_NAMES, contributions(d, protocols[p], preds[p])[0].tolist()))
                                for p in protocols}
        for p in protocols:
            np.savez(OUT / f"ens_{feat_set}_{ens.replace(' ', '_').replace('+', '-')}_{p}_preds.npz", *preds[p])
        summary[ens] = res
        print(fmt(res), flush=True)
        for p in protocols:
            print(f"    {p:6s} contrib: " + "  ".join(f"{k} {v:.2f}" for k, v in res["contrib_plain"][p].items()),
                  flush=True)
    (OUT / f"ensemble_{feat_set}.json").write_text(json.dumps(summary, indent=2))
