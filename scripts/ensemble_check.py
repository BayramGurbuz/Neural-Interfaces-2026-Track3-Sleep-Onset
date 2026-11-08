"""Ensemble check: mean of c3/c4/c5 OOF predictions, then leak-free calibration, vs c5 alone.

Usage: python scripts/ensemble_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_oof import calibrated
from cv_utils import N_FOLDS, causal_smooth
from stream_utils import RANGE_NAMES, summarize

G = Path.home() / "neuralbench" / "save" / "cv" / "grid"
MEMBERS = ("c3", "c4", "c5")

oofs = {c: pd.read_parquet(G / f"{c}_oof.parquet") for c in MEMBERS}
d = oofs["c5"].reset_index(drop=True)
for c, o in oofs.items():
    assert (o.timeline.to_numpy() == d.timeline.to_numpy()).all() and (o.stop.to_numpy() == d.stop.to_numpy()).all()
fold = d.fold.to_numpy()
variants = {"c5": d.pred.to_numpy(), "mean(c3,c4,c5)": np.mean([oofs[c].pred.to_numpy() for c in MEMBERS], axis=0)}
res = {}
for name, p in variants.items():
    for step, q in (("plain", p), ("calib", calibrated(d, fold, p, 20))):
        for sm in (0.0, 0.3):
            qq = causal_smooth(d, q, sm)
            sc = [summarize(d[fold == k], qq[fold == k]) for k in range(N_FOLDS)]
            res[(name, step, sm)] = dict(folds=np.array([s["wbmae"] for s in sc]),
                                         **{r: float(np.mean([s[r] for s in sc])) for r in RANGE_NAMES})
print(f"{'model':16s} {'step':6s} {'a':>4s} {'CV mean ± std':>15s}  {'diff vs c5 (same step/a)':>26s}  [0,40)")
out = {}
for (name, step, sm), r in res.items():
    f = r["folds"]
    diff = f - res[("c5", step, sm)]["folds"]
    print(f"{name:16s} {step:6s} {sm:4.1f} {f.mean():7.2f} ± {f.std(ddof=1):4.2f}   "
          f"{diff.mean():+.3f} ({(diff < 0).sum()}/5 better) {np.round(diff, 2).tolist()}  {r['[0,40)']:.1f}")
    out[f"{name}|{step}|{sm}"] = dict(folds=f.tolist(), mean=f.mean(), std=f.std(ddof=1), diff_vs_c5=diff.tolist(),
                                      **{k: r[k] for k in RANGE_NAMES})
(G / "ensemble_check.json").write_text(json.dumps(out, indent=2, default=float))
