"""Streaming (causal, per recording) summaries of the N2-detector output P(N2); the single source for training
(n2det_feats.summaries) and for the submission wrapper (embedded verbatim by build_hgb_submission.py).

Per window, from the scalar P of the current window: [P, EMA 30 s, EMA 2 min, running max of EMA 30 s,
EMA 30 s minus its value 30 s (RISE_LAG windows) earlier (0 before that)]. EMAs start at the first value;
alpha = 1 - exp(-step / tau). Arithmetic in float64, as the training tables (cast to float32 by the caller).
"""
import numpy as np

N2_TAUS = (30.0, 120.0)
RISE_LAG = 6
N2_NAMES = ("p", "ema30", "ema120", "runmax30", "rise30")


class N2Summary:
    def __init__(self, step_s=5.0):
        self.alphas = [1.0 - np.exp(-step_s / t) for t in N2_TAUS]
        self.reset()

    def reset(self):
        self.emas = None
        self.runmax = -np.inf
        self.hist = []  # EMA 30 s values, oldest first (at most RISE_LAG + 1 kept)

    def update(self, p):
        """p: finite scalar P(N2) of the current window -> (5,) float64."""
        p = float(p)
        if self.emas is None:
            self.emas = [p for _ in self.alphas]
        else:
            self.emas = [e + a * (p - e) for e, a in zip(self.emas, self.alphas)]
        e30 = self.emas[0]
        self.runmax = max(self.runmax, e30)
        self.hist.append(e30)
        rise = e30 - self.hist[-1 - RISE_LAG] if len(self.hist) > RISE_LAG else 0.0
        if len(self.hist) > RISE_LAG + 1:
            del self.hist[0]
        return np.array([p, e30, self.emas[1], self.runmax, rise])
