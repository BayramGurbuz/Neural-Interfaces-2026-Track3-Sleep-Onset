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
Optional groups (default off; the default output is unchanged):
- window level, ``window_features_ext``: "spec" (alpha/beta peak frequency, spectral edge 95 %,
  aperiodic log-log slope 2-30 Hz, spectral entropy 0.5-30 Hz, per channel), "eye" (log power
  0.2-0.6 Hz of AF7-AF8, blink count on mean(AF7, AF8) with a per-window adaptive threshold).
- stream level, ``CausalFeatureStream(groups=...)``: "dyn" (on the core band features: causal
  least-squares slopes over the last 30 s / 1 / 2 / 5 min, 5 min variance and lag-1
  autocorrelation, 10 min EMA), "maxmin" (30 s EMA minus its running max and running min),
  "long" (core features: causal slopes over 10 / 20 min, 10 min mean minus the baseline, 20 min EMA).
Nothing depends on future windows, the recording length or its end.
"""
import numpy as np
from scipy.signal import find_peaks, welch

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


SPEC = ["alpha_pf", "beta_pf", "sef95", "aper_slope", "spec_ent"]
CORE = PER_CHANNEL[:7]  # band powers + ratios: inputs of the "dyn" group
DYN_WINDOWS_S = (30.0, 60.0, 120.0, 300.0)
DYN_VAR_S = 300.0
DYN_EMA_S = 600.0
LONG_WINDOWS_S = (600.0, 1200.0)  # "long" group: 10 / 20 min causal slopes on the core features
LONG_EMA_S = 1200.0


def window_features_ext(X, sfreq, ch_names, groups):
    """Extra per-window features for the "spec" / "eye" groups -> (B, n_extra)."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(axis=-1, keepdims=True)
    out = []
    if "spec" in groups:
        freqs, psd = welch(X, fs=sfreq, nperseg=min(int(2 * sfreq), X.shape[-1]), axis=-1)
        def peak(lo, hi):
            m = (freqs >= lo) & (freqs <= hi)
            return freqs[m][np.argmax(psd[..., m], axis=-1)]
        m = (freqs >= 0.5) & (freqs <= 30)
        p = psd[..., m] + 1e-12
        cum = np.cumsum(p, axis=-1) / p.sum(axis=-1, keepdims=True)
        sef = freqs[m][np.minimum((cum < 0.95).sum(axis=-1), m.sum() - 1)]
        pn = p / p.sum(axis=-1, keepdims=True)
        ent = -(pn * np.log(pn)).sum(axis=-1) / np.log(m.sum())
        ma = (freqs >= 2) & (freqs <= 30)
        lf = np.log10(freqs[ma])
        lp = np.log10(psd[..., ma] + 1e-12)
        slope = ((lf - lf.mean()) * (lp - lp.mean(axis=-1, keepdims=True))).sum(axis=-1) / ((lf - lf.mean()) ** 2).sum()
        spec = np.stack([peak(7, 13), peak(15, 30), sef, slope, ent], axis=-1)  # (B, C, 5)
        out.append(spec.reshape(spec.shape[0], -1))
    if "eye" in groups:
        i7, i8 = find_channel(ch_names, "AF7"), find_channel(ch_names, "AF8")
        bip = X[:, i7] - X[:, i8]
        bip = bip - bip.mean(axis=-1, keepdims=True)
        spec = np.abs(np.fft.rfft(bip * np.hanning(bip.shape[-1]), axis=-1)) ** 2
        f = np.fft.rfftfreq(bip.shape[-1], 1.0 / sfreq)
        sem = np.log10(spec[:, (f >= 0.2) & (f <= 0.6)].sum(axis=-1) + EPS)
        front = 0.5 * (X[:, i7] + X[:, i8])
        blinks = []
        for x in front:
            mad = 1.4826 * np.median(np.abs(x - np.median(x)))
            pk, _ = find_peaks(np.abs(x), height=max(50.0, 6.0 * mad), distance=max(1, int(0.3 * sfreq)))
            blinks.append(len(pk))
        out.append(np.stack([sem, np.array(blinks, dtype=np.float64)], axis=-1))
    if not out:
        return np.zeros((X.shape[0], 0))
    return np.concatenate(out, axis=-1)


def find_channel(ch_names, name):
    """Index of channel ``name``: exact match, else case-insensitive match ignoring prefixes like "EEG "."""
    names = list(ch_names)
    if name in names:
        return names.index(name)
    norm = [str(n).upper().replace("EEG", "").strip(" -_") for n in names]
    return norm.index(name.upper())


def ext_names(ch_names, groups):
    names = []
    if "spec" in groups:
        names += [f"{ch}_{f}" for ch in ch_names for f in SPEC]
    if "eye" in groups:
        names += ["AF7-AF8_sem_0.2-0.6Hz", "blink_count"]
    return names


def core_index(ch_names):
    return [i * len(PER_CHANNEL) + j for i in range(len(ch_names)) for j in range(len(CORE))]


def feature_names(ch_names, groups=(), window_names=None):
    """window_names: names of the per-window vector (base + window-level extras)."""
    base = window_names if window_names is not None else [f"{ch}_{f}" for ch in ch_names for f in PER_CHANNEL]
    cur = base + [f"{n}_ema{int(t)}s" for t in EMA_TAUS_S for n in base]
    names = cur + [f"{n}_rel" for n in cur] + ["elapsed_s"]
    core = [base[i] for i in core_index(ch_names)]
    if "dyn" in groups:
        names += [f"{n}_slope{int(w)}s" for w in DYN_WINDOWS_S for n in core]
        names += [f"{n}_var{int(DYN_VAR_S)}s" for n in core] + [f"{n}_ac1_{int(DYN_VAR_S)}s" for n in core]
        names += [f"{n}_ema{int(DYN_EMA_S)}s" for n in core]
    if "maxmin" in groups:
        e30 = [f"{n}_ema{int(EMA_TAUS_S[0])}s" for n in base]
        names += [f"{n}_minus_runmax" for n in e30] + [f"{n}_minus_runmin" for n in e30]
    if "long" in groups:
        names += [f"{n}_slope{int(w)}s" for w in LONG_WINDOWS_S for n in core]
        names += [f"{n}_mean{int(LONG_WINDOWS_S[0])}s_rel" for n in core]
        names += [f"{n}_ema{int(LONG_EMA_S)}s" for n in core]
    return names


class CausalFeatureStream:
    """Stateful causal feature builder for one recording; ``reset()`` before each recording."""

    def __init__(self, step_s, groups=(), core_idx=None):
        self.step_s = float(step_s)
        self.alphas = [1.0 - np.exp(-self.step_s / tau) for tau in EMA_TAUS_S]
        self.groups = tuple(groups)
        self.core_idx = np.asarray(core_idx if core_idx is not None else [], dtype=int)
        self.dyn_n = [max(1, int(round(w / self.step_s))) for w in DYN_WINDOWS_S]
        self.var_n = max(1, int(round(DYN_VAR_S / self.step_s)))
        self.long_n = [max(2, int(round(w / self.step_s))) for w in LONG_WINDOWS_S]
        self.hist_n = max(self.dyn_n + [self.var_n] + (self.long_n if "long" in self.groups else []))
        self.alpha_long = 1.0 - np.exp(-self.step_s / DYN_EMA_S)
        self.alpha_vlong = 1.0 - np.exp(-self.step_s / LONG_EMA_S)
        self.reset()

    def reset(self):
        self.n_seen = 0
        self.emas = None
        self.base_sum = None
        self.base_n = 0
        self.hist = []
        self.ema_long = None
        self.ema_vlong = None
        self.run_max = None
        self.run_min = None

    def _push(self, c):
        """Append the newest core vector to the causal history (last ``hist_n`` windows)."""
        self.hist.append(c)
        if len(self.hist) > self.hist_n:
            self.hist.pop(0)

    def _long(self, f, base):
        """"long" group: slopes over LONG_WINDOWS_S, mean of the last LONG_WINDOWS_S[0] minus the
        recording baseline, EMA with LONG_EMA_S; all on the core features, history only."""
        c = f[self.core_idx]
        if self.ema_vlong is None:
            self.ema_vlong = c.copy()
        else:
            self.ema_vlong += self.alpha_vlong * (c - self.ema_vlong)
        H = np.asarray(self.hist)
        out = []
        for n in self.long_n:
            h = H[-n:]
            if len(h) < 2:
                out.append(np.zeros_like(c))
                continue
            t = np.arange(len(h), dtype=np.float64) * self.step_s
            t -= t.mean()
            out.append((t[:, None] * (h - h.mean(axis=0))).sum(axis=0) / (t ** 2).sum())
        out.append(H[-self.long_n[0]:].mean(axis=0) - base[self.core_idx])
        out.append(self.ema_vlong.copy())
        return out

    def _dyn(self, f):
        c = f[self.core_idx]
        if self.ema_long is None:
            self.ema_long = c.copy()
        else:
            self.ema_long += self.alpha_long * (c - self.ema_long)
        H = np.asarray(self.hist)
        out = []
        for n in self.dyn_n:
            h = H[-n:]
            if len(h) < 2:
                out.append(np.zeros_like(c))
                continue
            t = np.arange(len(h), dtype=np.float64) * self.step_s
            t -= t.mean()
            out.append((t[:, None] * (h - h.mean(axis=0))).sum(axis=0) / (t ** 2).sum())
        h = H[-self.var_n:]
        out.append(h.var(axis=0))
        if len(h) >= 3:
            a, b = h[:-1] - h[:-1].mean(axis=0), h[1:] - h[1:].mean(axis=0)
            den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
            out.append(np.where(den > 0, (a * b).sum(axis=0) / np.where(den > 0, den, 1.0), 0.0))
        else:
            out.append(np.zeros_like(c))
        out.append(self.ema_long.copy())
        return out

    def _maxmin(self):
        e = self.emas[0]
        if self.run_max is None:
            self.run_max, self.run_min = e.copy(), e.copy()
        else:
            np.maximum(self.run_max, e, out=self.run_max)
            np.minimum(self.run_min, e, out=self.run_min)
        return [e - self.run_max, e - self.run_min]

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
        parts = cur + [c - base for c in cur] + [np.array([elapsed])]
        if "dyn" in self.groups or "long" in self.groups:
            self._push(f[self.core_idx])
        if "dyn" in self.groups:
            parts += self._dyn(f)
        if "maxmin" in self.groups:
            parts += self._maxmin()
        if "long" in self.groups:
            parts += self._long(f, base)
        return np.concatenate(parts)
