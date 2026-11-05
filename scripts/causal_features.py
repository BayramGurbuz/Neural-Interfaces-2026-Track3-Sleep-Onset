"""Causal per-window features for Track 03 (numpy + scipy only, copied verbatim into submission.py).

Two layers:
- ``window_features(X, sfreq)``: features of each window on its own (no state).
  Per channel: log band powers (delta 0.5-4, theta 4-8, alpha 8-12, sigma 12-15,
  beta 15-30 Hz, Welch PSD), log theta/alpha, log (delta+theta)/(alpha+beta),
  log variance, log line length, log max |x| (window de-meaned first).
- ``CausalFeatureStream``: consumes those vectors one window at a time, in time order,
  and adds causal exponential moving averages (30 s, 2 min, 5 min), the difference of each
  of these to the recording's own baseline (mean over its first 2 min; running mean until
  2 min have passed) and the elapsed time since the recording started.
Nothing depends on future windows, the recording length or its end.
"""
import numpy as np
from scipy.signal import welch

BANDS = (("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0),
         ("sigma", 12.0, 15.0), ("beta", 15.0, 30.0))
PER_CHANNEL = [b[0] for b in BANDS] + ["theta_alpha", "slow_fast", "var", "linelen", "maxabs"]
EMA_TAUS_S = (30.0, 120.0, 300.0)
BASELINE_S = 120.0
EPS = 1e-6


def window_features(X, sfreq):
    """X: (B, C, T) array in microvolts -> (B, C * len(PER_CHANNEL)) float64, channel-major."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(axis=-1, keepdims=True)
    n_times = X.shape[-1]
    freqs, psd = welch(X, fs=sfreq, nperseg=min(int(2 * sfreq), n_times), axis=-1)
    df = freqs[1] - freqs[0]
    bp = {name: psd[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df for name, lo, hi in BANDS}
    feats = [np.log10(bp[name] + EPS) for name, _, _ in BANDS]
    feats.append(np.log10(bp["theta"] + EPS) - np.log10(bp["alpha"] + EPS))
    feats.append(np.log10(bp["delta"] + bp["theta"] + EPS) - np.log10(bp["alpha"] + bp["beta"] + EPS))
    feats.append(np.log10(X.var(axis=-1) + EPS))
    feats.append(np.log10(np.abs(np.diff(X, axis=-1)).mean(axis=-1) + EPS))
    feats.append(np.log10(np.abs(X).max(axis=-1) + EPS))
    out = np.stack(feats, axis=-1)  # (B, C, F)
    return out.reshape(out.shape[0], -1)


def feature_names(ch_names):
    base = [f"{ch}_{f}" for ch in ch_names for f in PER_CHANNEL]
    cur = base + [f"{n}_ema{int(t)}s" for t in EMA_TAUS_S for n in base]
    return cur + [f"{n}_rel" for n in cur] + ["elapsed_s"]


class CausalFeatureStream:
    """Stateful causal feature builder for one recording; ``reset()`` before each recording."""

    def __init__(self, step_s):
        self.step_s = float(step_s)
        self.alphas = [1.0 - np.exp(-self.step_s / tau) for tau in EMA_TAUS_S]
        self.reset()

    def reset(self):
        self.n_seen = 0
        self.emas = None
        self.base_sum = None
        self.base_n = 0

    def update(self, f):
        """f: (F,) window features of the newest window -> full causal feature vector."""
        self.n_seen += 1
        elapsed = self.n_seen * self.step_s
        if self.emas is None:
            self.emas = [f.copy() for _ in self.alphas]
            self.base_sum = np.zeros_like(f)
        else:
            for ema, a in zip(self.emas, self.alphas):
                ema += a * (f - ema)
        if elapsed <= BASELINE_S + 1e-9 or self.base_n == 0:
            self.base_sum += f
            self.base_n += 1
        base = self.base_sum / self.base_n
        cur = [f] + self.emas
        return np.concatenate(cur + [c - base for c in cur] + [np.array([elapsed])])
