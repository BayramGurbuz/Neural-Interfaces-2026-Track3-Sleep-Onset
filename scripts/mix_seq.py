"""Mix one or more sequence models (GRU / TCN / GRU+CNN ...: gru_<name>_{proto}_proba.npz, B-class
probabilities) with the HGB ensemble (c) on the common grid, dual CV.

Mix = alpha * q + (1 - alpha) * Wh; q = mean over the models of their metric-weighted, normalised grid
distributions (as the original (GRU+TCN)/2 run), Wh as mix_calib.components;
decision = continuous weighted median with omega = 1. alpha per fold chosen nested-LOFO: the loss is
summed over the other 9 folds (both protocols) with the held-out fold's recordings masked.
Reports the range breakdown against mix_uw_n10 (= mix_gru10_hgb, 40.28) and optional other references.

Usage: python scripts/mix_seq.py OUT_NAME SEQ_NAME [SEQ_NAME ...] [--ref NAME ...]
  e.g. python scripts/mix_seq.py mix_grutcn_n10 uw_n10 tcn_n10
       python scripts/mix_seq.py mix_cnn3 cnn3_emb --ref mix_uw_n3
"""
import sys

import numpy as np

from dist_hgb import VARIANTS, class_stats, decide
from dual_cv import OUT, load_trainval, seen_splits, unseen_splits
from ensemble_dist import GRID, MEMBERS, grid_omega, to_grid
from mix_calib import nested_select, to_protocols
from range_breakdown import breakdown, load_preds, print_breakdown

FEAT_SET = "dyn_maxmin_spec_eye"
HGB_MEMBERS = ["B", "B_h1", "B_h2", "B2"]
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
ONES = np.ones(len(GRID) - 1)


def components(d, protocols, seq_names):
    hgb = {m: np.load(OUT / f"ens_{FEAT_SET}_{m}.npz", allow_pickle=True)["probas"].item() for m in HGB_MEMBERS}
    folds = []
    for proto, splits in protocols.items():
        zs = [np.load(OUT / f"gru_{n}_{proto}_proba.npz") for n in seq_names]
        for k, (tr, te) in enumerate(splits):
            dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
            Wh = np.mean([to_grid(hgb[m][proto][k], MEMBERS[m][0]) for m in HGB_MEMBERS], axis=0) * grid_omega(dtr)
            Wh /= Wh.sum(1, keepdims=True)
            _, ob = class_stats(dtr, VARIANTS["B"])
            qs = []
            for z in zs:  # each model: metric-weighted, normalised, on the grid; then averaged
                q = z[f"arr_{k}"].astype(np.float64) * ob
                qs.append(to_grid(q / q.sum(1, keepdims=True), VARIANTS["B"]))
            q = np.mean(qs, axis=0)
            folds.append(dict(proto=proto, k=k, q=q, Wh=Wh, dte=dte, dtr=dtr))
    return folds


if __name__ == "__main__":
    args = sys.argv[1:]
    refs = args[args.index("--ref") + 1:] if "--ref" in args else []
    args = args[:args.index("--ref")] if "--ref" in args else args
    out_name, seq_names = args[0], args[1:]
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    folds = components(d, protocols, seq_names)
    cand = [np.stack([decide(a * f["q"] + (1 - a) * f["Wh"], None, ONES, True, GRID) for a in ALPHAS]) for f in folds]
    regions = [np.zeros(len(f["dte"]), int) for f in folds]
    preds, choices = nested_select(folds, cand, regions, 1)
    P = to_protocols(folds, preds)
    for p in protocols:
        np.savez(OUT / f"{out_name}_{p}_preds.npz", *P[p])
    b = breakdown(d, protocols, P)
    print(f"alphas {[float(ALPHAS[c[0]]) for c in choices]}")
    for ref in ["mix_uw_n10"] + refs:
        print_breakdown(f"{out_name} = mean({seq_names}) + HGB(c) vs {ref}", b,
                        breakdown(d, protocols, load_preds(ref, protocols)))
