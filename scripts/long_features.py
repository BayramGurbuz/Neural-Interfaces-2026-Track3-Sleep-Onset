"""Long-horizon feature group ("long": 10 / 20 min slopes, 10 min mean minus baseline, 20 min EMA on the
28 core features) with HGB member B, dual CV; compared with the cached B on dyn_maxmin_spec_eye (same
folds, same trees schedule), alone and mixed with GRU-uw (10 seeds, α chosen LOFO per protocol).
Member probabilities are cached as ens_<feat_set>_B.npz (ensemble_dist.member_probas).

Usage: python scripts/long_features.py [feat_set]   (default dyn_maxmin_spec_eye_long)
"""
import sys
import time

import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from dist_hgb import VARIANTS, class_stats, decide
from dual_cv import OUT, seen_splits, unseen_splits
from ensemble_dist import GRID, grid_omega, member_probas, to_grid
from range_breakdown import breakdown, print_breakdown
from stream_utils import summarize

B = VARIANTS["B"]
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
ONES = np.ones(len(GRID) - 1)
REF = "dyn_maxmin_spec_eye"


def own(d, protocols, probas):
    out = {}
    for proto, splits in protocols.items():
        out[proto] = []
        for (tr, _), P in zip(splits, probas[proto]):
            _, om = class_stats(d[tr].reset_index(drop=True), B)
            out[proto].append(decide(P.astype(np.float64), None, om, True, B).astype(np.float32))
    return out


def mix(d, protocols, probas, gru):
    out, alphas = {}, {}
    for proto, splits in protocols.items():
        cand, curves = [], []
        for (tr, te), P, G in zip(splits, probas[proto], gru[proto]):
            dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
            Wh = to_grid(P.astype(np.float64), B) * grid_omega(dtr)
            Wh /= Wh.sum(1, keepdims=True)
            _, ob = class_stats(dtr, B)
            q = G.astype(np.float64) * ob
            q = to_grid(q / q.sum(1, keepdims=True), B)
            c = [decide(a * q + (1 - a) * Wh, None, ONES, True, GRID).astype(np.float32) for a in ALPHAS]
            cand.append(c)
            curves.append([summarize(dte, x)["wbmae"] for x in c])
        C = np.array(curves)
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
        out[proto] = [cand[k][j] for k, j in enumerate(sel)]
        alphas[proto] = [float(ALPHAS[j]) for j in sel]
    return out, alphas


if __name__ == "__main__":
    feat_set = sys.argv[1] if len(sys.argv) > 1 else "dyn_maxmin_spec_eye_long"
    t0 = time.perf_counter()
    df, X, names = build_table_ext(*ABLATIONS[feat_set])
    m = (df.split != "test").to_numpy()
    d, X = df[m].reset_index(drop=True), X[m]
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    print(f"{feat_set}: {X.shape[1]} features (built in {time.perf_counter() - t0:.0f} s); "
          f"new: {[n for n in names if '_slope600s' in n or '_slope1200s' in n or 'mean600s' in n or 'ema1200s' in n][:4]} ...",
          flush=True)
    probas_new, n_iter = member_probas("B", feat_set, X, d, protocols)
    print(f"trees per fold {n_iter}", flush=True)
    probas_ref = np.load(OUT / f"ens_{REF}_B.npz", allow_pickle=True)["probas"].item()
    gru = {p: [np.load(OUT / f"gru_uw_n10_{p}_proba.npz")[f"arr_{k}"] for k in range(len(s))]
           for p, s in protocols.items()}

    pB = own(d, protocols, probas_ref)
    bB = breakdown(d, protocols, pB)
    print_breakdown(f"B on {REF} (ref)", bB)
    pL = own(d, protocols, probas_new)
    for p in protocols:
        np.savez(OUT / f"B_{feat_set}_{p}_preds.npz", *pL[p])
    print_breakdown(f"B on {feat_set}", breakdown(d, protocols, pL), bB)

    mB, aB = mix(d, protocols, probas_ref, gru)
    bmB = breakdown(d, protocols, mB)
    print_breakdown(f"B + GRU (ref), α {aB}", bmB)
    mL, aL = mix(d, protocols, probas_new, gru)
    for p in protocols:
        np.savez(OUT / f"mixB_{feat_set}_{p}_preds.npz", *mL[p])
    print_breakdown(f"B_{feat_set} + GRU, α {aL}", breakdown(d, protocols, mL), bmB)
