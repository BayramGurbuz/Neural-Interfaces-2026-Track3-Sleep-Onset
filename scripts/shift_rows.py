"""Work B / candidate 3: 40-test-recording scores of mix rows under the Work A shift distributions (the shifted dual CV's set)
and a robustness table (fixed k = 0, 1, 3, 6, 12, 24, 60 and wide random starts), from OUT/shift_calib_<row>.npz.
Δ against the first row (candidate 2).

Usage: python scripts/shift_rows.py [cand2 wb2 wb3 wb4 wb5]   -> table on stdout, OUT/shift_rows.json
"""
import json
import sys

import numpy as np

from dual_cv import OUT


def weights(name):
    w = np.zeros(61)
    if name.startswith("k"):
        w[int(name[1:])] = 1
    elif name.startswith("U"):
        w[:int(name[1:]) + 1] = 1
    else:  # m<p>U<K>: p * k0 + (1 - p) * U[0, K]
        p, K = name[1:].split("U")
        p, K = float(p), int(K)
        w[:K + 1] = (1 - p) / (K + 1)
        w[0] += p
    return w


def expected(S, w):
    W = np.where(np.isnan(S), 0.0, w[None, :])
    tot = W.sum(1)
    val = (np.nan_to_num(S) * W).sum(1) / np.where(tot > 0, tot, 1)
    last = np.array([row[~np.isnan(row)][-1] for row in S])
    return float(np.where(tot > 0, val, last).mean())


DISTS = ["k0", "k1", "k2", "k3", "m0.75U12", "m0.5U12", "U6", "U12", "m0.5U24"]
ROBUST = [0, 1, 3, 6, 12, 24, 60]

if __name__ == "__main__":
    rows = sys.argv[1:] or ["cand2", "wb2", "wb3", "wb4", "wb5"]
    Z = {r: np.load(OUT / f"shift_calib_{r}.npz") for r in rows}
    res = {}
    for r in rows:
        S = Z[r]["S"]
        res[r] = dict(dists={n: expected(S, weights(n)) for n in DISTS},
                      robust={**{f"k{k}": expected(S, weights(f"k{k}")) for k in ROBUST}, "random": float(Z[r]["S_rand"].mean())})
    for part, cols in (("dists", DISTS), ("robust", [f"k{k}" for k in ROBUST] + ["random"])):
        print(f"\n{part}: score (Δ vs {rows[0]})")
        print(f"{'row':8s} " + " ".join(f"{c:>15s}" for c in cols))
        for r in rows:
            print(f"{r:8s} " + " ".join(f"{res[r][part][c]:6.2f} ({res[r][part][c] - res[rows[0]][part][c]:+5.2f})" for c in cols))
    (OUT / "shift_rows.json").write_text(json.dumps(res, indent=2))
