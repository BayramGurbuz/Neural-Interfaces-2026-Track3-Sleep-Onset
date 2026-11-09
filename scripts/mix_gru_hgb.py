"""Step D: mix GRU class distributions with HGB ensemble (c), dual CV, alpha chosen leak-free (LOFO).

Both sides are turned into metric-weighted distributions on the common grid (ensemble_dist.GRID):
- HGB (c): mean of the 4 members' grid distributions, times the fold's grid omega, normalised.
- GRU: weighted-CE configs already learn a metric-weighted q; unweighted ones get q * omega_B
  (omega of the B classes from the fold's training windows), then spread uniformly onto the grid.
Mix = alpha * GRU + (1 - alpha) * HGB; decision = continuous weighted median with omega = 1.
alpha in ALPHAS; fold k uses the alpha minimising the mean curve of the other folds of the same protocol.
Reported plain + calibrated with per-bin contributions; fold-paired vs (c) and vs the GRU alone.

Usage: python scripts/mix_gru_hgb.py GRU_CONFIG [GRU_CONFIG ...]
"""
import json
import sys

import numpy as np

from dist_hgb import VARIANTS, class_stats, decide
from dual_cv import OUT, calibrate, dual_report, fmt, load_trainval, seen_splits, unseen_splits
from ensemble_dist import GRID, MEMBERS, grid_omega, to_grid
from gru_cv import CONFIGS
from report_bins import contributions
from stream_utils import RANGE_NAMES, summarize

FEAT_SET = "dyn_maxmin_spec_eye"
HGB_MEMBERS = ["B", "B_h1", "B_h2", "B2"]
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
B = VARIANTS["B"]


def fold_scores(d, splits, preds):
    return np.array([summarize(d[te].reset_index(drop=True), p)["wbmae"] for (_, te), p in zip(splits, preds)])


if __name__ == "__main__":
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    hgb = {m: np.load(OUT / f"ens_{FEAT_SET}_{m}.npz", allow_pickle=True)["probas"].item() for m in HGB_MEMBERS}
    summary_path = OUT / "mix_gru_hgb.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for cfg in sys.argv[1:]:
        weighted = CONFIGS[cfg]["weighted"]
        preds, alphas, gru_only, hgb_only = {}, {}, {}, {}
        for proto, splits in protocols.items():
            z = np.load(OUT / f"gru_{cfg}_{proto}_proba.npz")
            curves, cand = [], []
            for k, (tr, te) in enumerate(splits):
                dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
                og = grid_omega(dtr)
                Wh = np.mean([to_grid(hgb[m][proto][k], MEMBERS[m][0]) for m in HGB_MEMBERS], axis=0) * og
                Wh /= Wh.sum(1, keepdims=True)
                q = z[f"arr_{k}"].astype(np.float64)
                if not weighted:
                    _, ob = class_stats(dtr, B)
                    q = q * ob
                q = to_grid(q / q.sum(1, keepdims=True), B)
                c = [decide(a * q + (1 - a) * Wh, None, np.ones(len(GRID) - 1), True, GRID).astype(np.float32)
                     for a in ALPHAS]
                cand.append(c)
                curves.append([summarize(dte, x)["wbmae"] for x in c])
            C = np.array(curves)
            sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
            preds[proto] = [cand[k][j] for k, j in enumerate(sel)]
            alphas[proto] = [float(ALPHAS[j]) for j in sel]
            gru_only[proto] = [cand[k][-1] for k in range(len(splits))]
            hgb_only[proto] = [cand[k][0] for k in range(len(splits))]
            print(f"  {cfg} {proto}: mean curve over alpha " +
                  " ".join(f"{a:.1f}:{v:.2f}" for a, v in zip(ALPHAS, C.mean(0))), flush=True)
        name = f"GRU-{cfg}+HGB(c)"
        res = dual_report(name, d, protocols, preds, dict(alphas=alphas))
        res["contrib_plain"] = {p: dict(zip(RANGE_NAMES, contributions(d, protocols[p], preds[p])[0].tolist()))
                                for p in protocols}
        for p in protocols:
            np.savez(OUT / f"mix_{cfg}_{p}_preds.npz", *preds[p])
        summary[name] = res
        print(fmt(res), f"  alpha {alphas}", flush=True)
        for p in protocols:
            print(f"    {p:6s} contrib: " + "  ".join(f"{k} {v:.2f}" for k, v in res["contrib_plain"][p].items()),
                  flush=True)
        for ref_name, ref in (("HGB(c)", hgb_only), ("GRU alone", gru_only)):
            for cal in (False, True):
                tot, parts = 0, []
                for p, splits in protocols.items():
                    a, b = preds[p], ref[p]
                    if cal:
                        a, b = calibrate(d, splits, a), calibrate(d, splits, b)
                    diff = fold_scores(d, splits, a) - fold_scores(d, splits, b)
                    tot += int((diff < 0).sum())
                    parts.append(f"{p}: " + " ".join(f"{x:+.2f}" for x in diff))
                print(f"    vs {ref_name:9s} {'calib' if cal else 'plain'}: " + " | ".join(parts) +
                      f"  -> better {tot}/10", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2))
