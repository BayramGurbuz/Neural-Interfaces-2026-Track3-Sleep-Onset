"""Per-bin contribution to W-bMAE (w_b * MAE_b / sum_w per recording, averaged) for dual-CV runs,
plus a 30 s-grid robustness check: scores on recordings whose N2 onset lies on the 30 s grid vs
the 77 off-grid ones (all with onset > 900 s, probably cropped starts).

The onset grid flag is read from the session tables for this offline analysis only; it is never a
model input.

Usage: python scripts/report_bins.py name[:calib] ...   e.g. c5:calib dist_continuous B2_dist_continuous
"""
import glob
import sys

import numpy as np
import pandas as pd

from dual_cv import OUT, calibrate, load_trainval, seen_splits, unseen_splits
from stream_utils import RANGE_NAMES, WEIGHTS, per_recording_scores

DATA = "/home/bayram/neuralbench/data/Interaxon2026Muse/download/nm000287"


def grid_flags():
    rows = []
    for f in glob.glob(f"{DATA}/sub-*/sub-*_sessions.tsv"):
        sub = f.split("/")[-2]
        for _, r in pd.read_csv(f, sep="\t").iterrows():
            on = r.n2_onset % 30
            rows.append((f"Interaxon2026Muse:session={r.session_id},subject={sub}",
                         bool(np.isclose(on, 0) or np.isclose(on, 30)), r.n2_onset > 900))
    return pd.DataFrame(rows, columns=["timeline", "on_grid", "long"]).set_index("timeline")


def contributions(d, splits, preds):
    per_fold, recs = [], []
    for (_, te), p in zip(splits, preds):
        rec = per_recording_scores(d[te].reset_index(drop=True), p)
        mae = rec[RANGE_NAMES].to_numpy()
        w = np.where(np.isnan(mae), 0.0, WEIGHTS)
        per_fold.append((np.nan_to_num(mae) * w / w.sum(axis=1, keepdims=True)).mean(axis=0))
        recs.append(rec.wbmae)
    return np.mean(per_fold, axis=0), pd.concat(recs)


if __name__ == "__main__":
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    flags = grid_flags()
    for arg in sys.argv[1:]:
        name, _, variant = arg.partition(":")
        for proto, splits in protocols.items():
            z = np.load(OUT / f"{name}_{proto}_preds.npz")
            preds = [z[f"arr_{k}"] for k in range(len(splits))]
            if variant == "calib":
                preds = calibrate(d, splits, preds)
            c, rec = contributions(d, splits, preds)
            f = flags.loc[rec.index]
            g_on = rec[f.on_grid.to_numpy()].mean()
            g_off = rec[~f.on_grid.to_numpy()].mean()
            g_long_on = rec[(f.on_grid & f.long).to_numpy()].mean()
            print(f"{arg:28s} {proto:6s} W-bMAE {c.sum():6.2f} | " +
                  "  ".join(f"{r} {c[i]:5.2f} ({100 * c[i] / c.sum():2.0f}%)" for i, r in enumerate(RANGE_NAMES)) +
                  f" | grid: on {g_on:.1f}, off {g_off:.1f}, on & onset>900 s {g_long_on:.1f}")
