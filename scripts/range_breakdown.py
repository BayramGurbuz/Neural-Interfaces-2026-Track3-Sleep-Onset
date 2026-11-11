"""Per-range error breakdown of dual-CV OOF predictions, comparable to the sealed leaderboard columns.

For each protocol (unseen, seen) and for the macro average (unseen + seen) / 2:
- W-bMAE (fold mean, the CV decision number) and per-range contributions to it
  (per recording w_b * MAE_b / sum of present w, averaged over recordings and folds; they sum to W-bMAE);
- per-range MAE (mean over recordings having that range);
- bMAE / MAE with the warm-up/leaderboard definition: all held-out windows of the protocol pooled,
  bMAE = plain mean of the 4 range MAEs, MAE = plain mean absolute error.

Usage: python scripts/range_breakdown.py NAME [NAME ...]   (reads OUT/NAME_{unseen,seen}_preds.npz)
"""
import sys

import numpy as np

from dual_cv import OUT, load_trainval, seen_splits, unseen_splits
from report_bins import contributions
from stream_utils import RANGE_NAMES, range_index, summarize

SEALED_FINAL540 = dict(wbmae_macro=39.56, wbmae=39.91, bmae=101.36, mae=198.26)


def breakdown(d, protocols, preds_by_proto):
    """preds_by_proto: {proto: list of per-fold predictions}. Returns nested dict of numbers."""
    out = {}
    for proto, splits in protocols.items():
        preds = preds_by_proto[proto]
        contrib, _ = contributions(d, splits, preds)
        fold = [summarize(d[te].reset_index(drop=True), p) for (_, te), p in zip(splits, preds)]
        y = np.concatenate([d.target.to_numpy()[te] for _, te in splits])
        p = np.concatenate([np.asarray(x, float) for x in preds])
        err = np.abs(p - y)
        rng = range_index(y)
        pooled = np.array([err[rng == b].mean() for b in range(4)])
        out[proto] = dict(
            wbmae=float(np.mean([f["wbmae"] for f in fold])),
            folds=[f["wbmae"] for f in fold],
            contrib=contrib.tolist(),
            range_mae=[float(np.nanmean([f[r] for f in fold])) for r in RANGE_NAMES],
            pooled_range_mae=pooled.tolist(),
            bmae=float(pooled.mean()),
            mae=float(err.mean()),
        )
    out["macro"] = {k: (np.array(out["unseen"][k]) + np.array(out["seen"][k])) / 2
                    for k in ("wbmae", "contrib", "range_mae", "pooled_range_mae", "bmae", "mae")}
    return out


def print_breakdown(name, b, ref=None):
    print(f"== {name}")
    hdr = "  ".join(f"{r:>10s}" for r in RANGE_NAMES)
    print(f"  {'':22s} {'W-bMAE':>7s}  {hdr}   {'bMAE':>7s} {'MAE':>7s}")
    for proto in ("unseen", "seen", "macro"):
        r = b[proto]
        print(f"  {proto:6s} contribution    {float(r['wbmae']):7.2f}  " +
              "  ".join(f"{c:10.2f}" for c in r["contrib"]) + f"   {float(r['bmae']):7.2f} {float(r['mae']):7.2f}")
        print(f"  {proto:6s} range MAE (rec) {'':7s}  " + "  ".join(f"{c:10.2f}" for c in r["range_mae"]))
        print(f"  {proto:6s} range MAE (pool){'':7s}  " + "  ".join(f"{c:10.2f}" for c in r["pooled_range_mae"]))
    if ref is not None:
        dc = np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])
        print(f"  Δ macro vs ref        {float(b['macro']['wbmae'] - ref['macro']['wbmae']):+7.2f}  " +
              "  ".join(f"{c:+10.2f}" for c in dc) +
              f"   {float(b['macro']['bmae'] - ref['macro']['bmae']):+7.2f} {float(b['macro']['mae'] - ref['macro']['mae']):+7.2f}")
        for proto in ("unseen", "seen"):
            diff = np.array(b[proto]["folds"]) - np.array(ref[proto]["folds"])
            print(f"    {proto:6s} folds Δ: " + " ".join(f"{x:+.2f}" for x in diff))
        n = sum(int((np.array(b[p]["folds"]) < np.array(ref[p]["folds"])).sum()) for p in ("unseen", "seen"))
        print(f"    better in {n}/10 folds")


def load_preds(name, protocols):
    return {p: [np.load(OUT / f"{name}_{p}_preds.npz")[f"arr_{k}"] for k in range(len(s))]
            for p, s in protocols.items()}


if __name__ == "__main__":
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    ref = breakdown(d, protocols, load_preds("mix_uw_n10", protocols))
    for name in sys.argv[1:]:
        b = breakdown(d, protocols, load_preds(name, protocols))
        print_breakdown(name, b, None if name == "mix_uw_n10" else ref)
    print(f"sealed final540: {SEALED_FINAL540}")
