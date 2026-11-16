"""Work A analysis: which stream-start shift distribution best explains the sealed score DIFFERENCES of the 5 submissions?

Input: OUT/shift_calib_<model>.npz (shift_calib.py): per test recording, official W-bMAE with the stream started at window k
(k = 0..60) and at 3 wide random starts (test recordings of 12 / 42 / 42 / 48 windows: a fixed k beyond a recording's
last window uses its largest available k). A distribution over k gives each model's expected test score = mean over recordings of
the k-expectation of the recording's score (k beyond a recording's last window: that recording's available k only).
Fit: sealed = offset + test (slope 1, offset free: the sealed set may be easier); residual RMSE over the 5 models, Spearman rank
correlation, and the key differences next to the sealed ones. 40 recordings, seen subjects only -> direction, not proof.

Usage: python scripts/shift_fit.py   -> OUT/shift_fit.json, table on stdout
"""
import json

import numpy as np
from scipy.stats import spearmanr

from dual_cv import OUT

NAMES = ["final540", "mix3_540", "gridpi", "cand1", "cand2"]
SEALED = np.array([38.74, 37.79, 39.20, 37.09, 34.41])
PAIRS = [("cand1", "cand2"), ("mix3_540", "gridpi"), ("mix3_540", "cand1"), ("final540", "mix3_540")]
FIXED = [0, 1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 60]
UNIF = [6, 12, 24, 60]

Z = {m: np.load(OUT / f"shift_calib_{m}.npz") for m in NAMES}


def expected(m, w):
    """w: (61,) weights over k = 0..60 (renormalised per recording over the available k)."""
    S = Z[m]["S"]
    W = np.where(np.isnan(S), 0.0, w[None, :])
    tot = W.sum(1)
    val = (np.nan_to_num(S) * W).sum(1) / np.where(tot > 0, tot, 1)
    last = np.array([row[~np.isnan(row)][-1] for row in S])  # fixed k beyond a short recording: its largest available k
    return float(np.where(tot > 0, val, last).mean())


def dists():
    out = {}
    for k in FIXED:
        w = np.zeros(61); w[k] = 1; out[f"k={k}"] = ("w", w)
    for K in UNIF:
        w = np.zeros(61); w[:K + 1] = 1; out[f"U[0,{K}]"] = ("w", w)
    out["wide random"] = ("rand", None)
    for p in (0.25, 0.5, 0.75):
        for K in (12, 24, 60):
            w = np.zeros(61); w[:K + 1] = (1 - p) / (K + 1); w[0] += p
            out[f"{p:.2f}·k0 + {1 - p:.2f}·U[0,{K}]"] = ("w", w)
        out[f"{p:.2f}·k0 + {1 - p:.2f}·wide"] = ("mixw", p)
    return out


def scores(kind, w):
    if kind == "w":
        return np.array([expected(m, w) for m in NAMES])
    wide = np.array([Z[m]["S_rand"].mean() for m in NAMES])
    if kind == "rand":
        return wide
    k0 = np.array([np.nanmean(Z[m]["S"][:, 0]) for m in NAMES])
    return w * k0 + (1 - w) * wide


if __name__ == "__main__":
    rows = []
    for name, (kind, w) in dists().items():
        T = scores(kind, w)
        off = float(np.mean(SEALED - T))
        res = SEALED - T - off
        rmse = float(np.sqrt(np.mean(res ** 2)))
        rho = float(spearmanr(T, SEALED).correlation)
        diffs = {f"{a}->{b}": float(T[NAMES.index(b)] - T[NAMES.index(a)]) for a, b in PAIRS}
        rows.append(dict(dist=name, test=T.tolist(), offset=off, rmse=rmse, spearman=rho, diffs=diffs, residuals=res.tolist()))
    sealed_d = {f"{a}->{b}": float(SEALED[NAMES.index(b)] - SEALED[NAMES.index(a)]) for a, b in PAIRS}
    rows.sort(key=lambda r: r["rmse"])
    hdr = " ".join(f"{n:>8s}" for n in NAMES)
    print(f"{'distribution':28s} {hdr} | offset  RMSE  rho | " + " ".join(f"{k:>16s}" for k in sealed_d))
    print(f"{'SEALED':28s} " + " ".join(f"{v:8.2f}" for v in SEALED) + f" | {'':17s} | " + " ".join(f"{v:+16.2f}" for v in sealed_d.values()))
    for r in rows:
        print(f"{r['dist']:28s} " + " ".join(f"{v:8.2f}" for v in r["test"]) + f" | {r['offset']:+6.2f} {r['rmse']:5.2f} {r['spearman']:+.2f} | "
              + " ".join(f"{v:+16.2f}" for v in r["diffs"].values()))
    (OUT / "shift_fit.json").write_text(json.dumps(dict(sealed=SEALED.tolist(), sealed_diffs=sealed_d, names=NAMES, rows=rows), indent=2))
