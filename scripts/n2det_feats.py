"""Work 2 step e: causal summaries of the N2-detector output P(N2) per window, fold-specific.

Per detector (default cnn only, see DETS): P, EMA 30 s, EMA 2 min (alpha = 1 - exp(-5 / tau), started at the first value, as B2Stream),
running max of EMA 30 s, EMA 30 s minus its value 30 s (6 windows) earlier (0 before that) -> 5 features.
Inputs: OUT/n2det_<det>_<proto>_f<k>.npy (n2det_cv.py --inner): P for all train+val stream rows of outer fold k
(held-out rows from the outer detector, training rows out-of-fold from inner folds).
"""
import numpy as np

from dual_cv import OUT

DETS = ("cnn",)  # CNN detector only: the HGB detector (+ bip features) would cost ~3.2 ms/window (> +10 % of gridpi)
NAMES = [f"n2{det}_{s}" for det in DETS for s in ("p", "ema30", "ema120", "runmax30", "rise30")]


def summaries(p, recs):
    """p (N,) per-window P(N2); recs: list of row-index arrays in time order -> (N, 5) float32."""
    out = np.zeros((len(p), 5), np.float32)
    a30, a120 = 1 - np.exp(-5 / 30), 1 - np.exp(-5 / 120)
    for r in recs:
        x = p[r].astype(np.float64)
        e30, e120 = np.empty_like(x), np.empty_like(x)
        e30[0] = e120[0] = x[0]
        for i in range(1, len(x)):
            e30[i] = e30[i - 1] + a30 * (x[i] - e30[i - 1])
            e120[i] = e120[i - 1] + a120 * (x[i] - e120[i - 1])
        rise = np.zeros_like(x)
        rise[6:] = e30[6:] - e30[:-6]
        out[r] = np.stack([x, e30, e120, np.maximum.accumulate(e30), rise], 1)
    return out


def fold_features(proto, k, recs, dets=DETS):
    return np.concatenate([summaries(np.load(OUT / f"n2det_{det}_{proto}_f{k}.npy"), recs) for det in dets], axis=1)
