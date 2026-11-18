"""Work D (18 Nov): which stream-start shift distribution explains the sealed score DIFFERENCES of the 8 submissions after the
Nov 16 sealed update (the opening minutes of every sealed recording are no longer delivered; elapsed time counts from the first
delivered window)?

Input: OUT/shift_calib120_<model>.npz (shift_calib.py --kmax=120): per public-test recording (40, out-of-sample for the train+val
counterparts, seen subjects only), official W-bMAE with the stream started at window k = 0..120 (k <= n_windows - 1).
Distributions (k in windows, 12 windows = 1 min): fixed M min (M in 1, 2, 3, 4, 5, 6, 8, 10), uniform [a, b] min over the windows
12a..12b (a < b from the same values; [0, b] rows for information).
Short recordings (onset before the shift): "drop" (primary) = a recording with no window left is not delivered; the (recording, k)
pairs are pooled, i.e. each recording is weighted by its survival probability under the distribution. "clamp" (secondary, the
Work A convention) = a shift beyond the last window uses the recording's largest k.
Fit: sealed = offset + test (slope 1, offset free) -> residual RMSE over the 8 models (differences only, not the level); also the
RMSE over the 4 crop-robust models (cand2, cand3, ctx_c2, ctx_c3), whose small differences matter for the next choices, and the
free-slope fit for information. Bootstrap over recordings (1000): how often each distribution is in the top 3 by the 8-model RMSE.

Usage: python scripts/shift_fit_d.py   -> OUT/shift_fit_d.json, tables on stdout
"""
import json
from itertools import combinations

import numpy as np

from dual_cv import OUT

NAMES = ["final540", "mix3_540", "gridpi", "cand1", "cand2", "wb5", "c2ctx", "c3ctx"]
LABEL = dict(final540="final540", mix3_540="mix3", gridpi="gridpi", cand1="aday1", cand2="cand2", wb5="cand3", c2ctx="ctx_c2",
             c3ctx="ctx_c3")
SEALED = np.array([87.99, 62.02, 64.96, 47.42, 34.41, 33.72, 33.12, 33.09])
ROBUST = [4, 5, 6, 7]
PAIRS = [("cand2", "final540"), ("cand2", "mix3_540"), ("mix3_540", "gridpi"), ("cand2", "cand1"), ("cand2", "wb5"),
         ("cand2", "c2ctx"), ("cand2", "c3ctx"), ("c2ctx", "c3ctx")]
MIN = [1, 2, 3, 4, 5, 6, 8, 10]
KM = 120

Z = {m: np.load(OUT / f"shift_calib120_{m}.npz") for m in NAMES}
S_ALL = np.stack([Z[m]["S"] for m in NAMES])  # (8, 40, 121)
N_WIN = Z["cand2"]["n_windows"]
VALID = ~np.isnan(S_ALL[4])
assert all((~np.isnan(S_ALL[i]) == VALID).all() for i in range(8))


def dists():
    out = {}
    for M in MIN:
        w = np.zeros(KM + 1); w[12 * M] = 1; out[f"{M} min"] = w
    for a, b in combinations([0] + MIN, 2):
        w = np.zeros(KM + 1); w[12 * a:12 * b + 1] = 1; out[f"U[{a},{b}] min"] = w / w.sum()
    out["k0 (no shift)"] = np.eye(KM + 1)[0]
    return out


def expected(w, recs=None, mode="drop"):
    """(8,) expected test score of each model under the shift weights w over k = 0..120."""
    S = S_ALL if recs is None else S_ALL[:, recs]
    V = VALID if recs is None else VALID[recs]
    if mode == "drop":
        W = np.where(V, w[None, :], 0.0)  # (R, K)
        return (np.nan_to_num(S) * W[None]).sum((1, 2)) / W.sum()
    last = np.array([np.where(v)[0][-1] for v in V])  # clamp: shift beyond the last window -> the last window's k
    out = np.zeros(S.shape[0])
    for r in range(S.shape[1]):
        ks = np.minimum(np.arange(KM + 1), last[r])
        out += (S[:, r, ks] * w[None, :]).sum(1)
    return out / S.shape[1]


def fit(T):
    off = float(np.mean(SEALED - T))
    res = SEALED - T - off
    rob = SEALED[ROBUST] - T[ROBUST]
    rob = rob - rob.mean()
    A = np.c_[np.ones(8), T]
    coef = np.linalg.lstsq(A, SEALED, rcond=None)[0]
    res_s = SEALED - A @ coef
    return dict(offset=off, rmse=float(np.sqrt(np.mean(res ** 2))), rmse_robust=float(np.sqrt(np.mean(rob ** 2))),
                slope=float(coef[1]), rmse_free=float(np.sqrt(np.mean(res_s ** 2))), residuals=res.tolist())


def table(mode):
    rows = []
    for name, w in dists().items():
        T = expected(w, mode=mode)
        kept = float((np.where(VALID, w[None, :], 0).sum(1)).sum())  # expected number of delivered recordings
        diffs = {f"{a}->{b}": float(T[NAMES.index(b)] - T[NAMES.index(a)]) for a, b in PAIRS}
        rows.append(dict(dist=name, test=T.tolist(), kept=kept, diffs=diffs, **fit(T)))
    rows.sort(key=lambda r: r["rmse"])
    return rows


def show(rows, mode, n=None):
    sd = {f"{a}->{b}": float(SEALED[NAMES.index(b)] - SEALED[NAMES.index(a)]) for a, b in PAIRS}
    print(f"\n=== mode {mode} (sorted by 8-model RMSE, slope 1) ===")
    print(f"{'distribution':16s} " + " ".join(f"{LABEL[m]:>8s}" for m in NAMES) + " | kept  off  RMSE  rob  slope | "
          + " ".join(f"{k.replace('->', '>'):>14s}" for k in sd))
    print(f"{'SEALED':16s} " + " ".join(f"{v:8.2f}" for v in SEALED) + f" | {'':27s} | " + " ".join(f"{v:+14.2f}" for v in sd.values()))
    for r in rows[:n]:
        print(f"{r['dist']:16s} " + " ".join(f"{v:8.2f}" for v in r["test"]) + f" | {r['kept']:4.1f} {r['offset']:+5.1f} {r['rmse']:5.2f} "
              f"{r['rmse_robust']:4.2f} {r['slope']:5.2f} | " + " ".join(f"{v:+14.2f}" for v in r["diffs"].values()))


def bootstrap(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    D = dists()
    names = list(D)
    top1, top3 = dict.fromkeys(names, 0), dict.fromkeys(names, 0)
    for _ in range(n):
        recs = rng.integers(0, S_ALL.shape[1], S_ALL.shape[1])
        rm = [fit(expected(D[k], recs))["rmse"] for k in names]
        order = np.argsort(rm)
        top1[names[order[0]]] += 1
        for i in order[:3]:
            top3[names[i]] += 1
    return {k: dict(top1=top1[k] / n, top3=top3[k] / n) for k in names}


if __name__ == "__main__":
    res = {}
    for mode in ("drop", "clamp"):
        res[mode] = table(mode)
        show(res[mode], mode, n=None if mode == "drop" else 12)
    bs = bootstrap()
    print("\n=== bootstrap over recordings (drop mode, 1000): top-1 / top-3 frequency ===")
    for k, v in sorted(bs.items(), key=lambda kv: -kv[1]["top3"])[:12]:
        print(f"{k:16s} top1 {v['top1']:.3f}  top3 {v['top3']:.3f}")
    res["bootstrap"] = bs
    res["sealed"] = SEALED.tolist()
    res["names"] = NAMES
    (OUT / "shift_fit_d.json").write_text(json.dumps(res, indent=2))
