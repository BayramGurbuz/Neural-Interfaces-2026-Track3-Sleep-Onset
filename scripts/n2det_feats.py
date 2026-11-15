"""Work 2 step e: causal summaries of the N2-detector output P(N2) per window, fold-specific.

Per detector (default cnn only, see DETS): P, EMA 30 s, EMA 2 min (alpha = 1 - exp(-5 / tau), started at the first value, as B2Stream),
running max of EMA 30 s, EMA 30 s minus its value 30 s (6 windows) earlier (0 before that) -> 5 features.
Inputs: OUT/n2det_<det>_<proto>_f<k>.npy (n2det_cv.py --inner): P for all train+val stream rows of outer fold k
(held-out rows from the outer detector, training rows out-of-fold from inner folds).
"""
import numpy as np

from dual_cv import OUT
from n2det_stream import N2_NAMES, N2Summary

DETS = ("cnn",)  # CNN detector only: the HGB detector (+ bip features) would cost ~3.2 ms/window (> +10 % of gridpi)
NAMES = [f"n2{det}_{s}" for det in DETS for s in N2_NAMES]


def summaries(p, recs):
    """p (N,) per-window P(N2); recs: list of row-index arrays in time order -> (N, 5) float32 (N2Summary, the same
    code the submission wrapper runs; checked identical to the earlier vectorised version)."""
    out = np.zeros((len(p), 5), np.float32)
    st = N2Summary(5.0)
    for r in recs:
        st.reset()
        for i in r:
            out[i] = st.update(p[i])
    return out


def fold_features(proto, k, recs, dets=DETS):
    return np.concatenate([summaries(np.load(OUT / f"n2det_{det}_{proto}_f{k}.npy"), recs) for det in dets], axis=1)


MC_SEEDS = (0, 1, 2)


def fold_features_mc(proto, k, recs, keys=("close",), seeds=MC_SEEDS):
    """Work 3: N2Summary of the multi-class detector's outputs (mean over seeds) for outer fold k (n2det_v2_inner.py files,
    out-of-fold for every row) -> (N, 5 * len(keys)) float32."""
    zs = [np.load(OUT / f"n2det_multi_s{s}_{proto}_f{k}.npz") for s in seeds]
    return np.concatenate([summaries(np.mean([z[key] for z in zs], axis=0), recs) for key in keys], axis=1)
