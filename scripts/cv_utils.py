"""Subject-grouped 5-fold CV over train+val recordings (test stays held out) + causal smoothing."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

N_FOLDS = 5
CV_SEED = 0


def make_folds(d: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = CV_SEED) -> np.ndarray:
    """Fold id (0..n_folds-1) per row; all recordings of a subject share one fold."""
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(d), -1)
    for k, (_, te) in enumerate(gkf.split(d, groups=d.subject.to_numpy())):
        fold[te] = k
    return fold


def causal_smooth(d: pd.DataFrame, pred: np.ndarray, a: float, dt: float = 5.0) -> np.ndarray:
    """s_t = a * max(s_{t-1} - dt, 0) + (1 - a) * p_t per recording (s_1 = p_1), in time order.

    Only past predictions are used, so this can run inside the streaming submission.
    """
    pred = np.asarray(pred, dtype=float)
    if a == 0:
        return pred.copy()
    out = np.empty_like(pred)
    order = np.lexsort((d.stop.to_numpy(), d.timeline.to_numpy()))
    tl = d.timeline.to_numpy()
    prev_tl, s = None, 0.0
    for i in order:
        if tl[i] != prev_tl:
            s, prev_tl = pred[i], tl[i]
        else:
            s = a * max(s - dt, 0.0) + (1.0 - a) * pred[i]
        out[i] = s
    return out
