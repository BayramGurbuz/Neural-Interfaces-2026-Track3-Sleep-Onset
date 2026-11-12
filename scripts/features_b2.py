"""B2 feature groups (numpy + scipy only, so they can be copied verbatim into submission.py).

Window level (each window on its own, no state; filters run inside the current window only):
- "bip"    : bipolar derivations AF7-TP9, AF8-TP10, AF8-TP9, AF7-TP10, TP9-TP10, AF7-AF8; per derivation the
             spectral features of the base set (5 log band powers, log theta/alpha, log slow/fast) and the
             "spec" group (alpha / beta peak frequency, SEF95, aperiodic slope, spectral entropy) = 12.
- "events" : event detectors on the 4 channels and the 6 derivations (10 signals):
             spindles (11-16 Hz band, Hilbert envelope > 2.5 x the window's median envelope for 0.5-2 s):
               count, total duration (s), log peak envelope (uV);
             slow waves / K-complex candidates (0.5-2 Hz band, negative-to-positive half waves 0.25-1 s long):
               count with peak-to-peak > 75 uV, log max peak-to-peak;
             eyes (mean of AF7 and AF8, 0.5-10 Hz): blink counts with two thresholds (height > max(50, 6 MAD), as
             the "eye" group, and > max(30, 4 MAD)), slow-eye-movement count (AF7-AF8, 0.2-0.6 Hz,
             zero crossings of the band signal with amplitude > 20 uV).
Stream level (B2Stream, history only; reset per recording):
- "bip" / "events" values enter as current value + causal EMA 30 s / 2 min, and the difference of these to the
  recording's first-2-min running mean (6 x n features), as approved for B2.
- "epoch"  : 30 s epochs aligned to the recording start (windows ending at 5..30 s = epoch 1, ...): phase of
             the window inside its epoch (index 0..5) and, on the 28 core features (band powers + ratios per
             channel), mean of the last completed epoch, mean of the 2 and 4 epochs before it, and trend
             (last completed epoch minus the mean of the 4 before) = 1 + 4 x 28.
Nothing depends on future windows, the recording length or its end.
"""
import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch

BIP_PAIRS = (("AF7", "TP9"), ("AF8", "TP10"), ("AF8", "TP9"), ("AF7", "TP10"), ("TP9", "TP10"), ("AF7", "AF8"))
B2_BANDS = (("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0), ("sigma", 12.0, 15.0), ("beta", 15.0, 30.0))
BIP_FEATS = [b[0] for b in B2_BANDS] + ["theta_alpha", "slow_fast", "alpha_pf", "beta_pf", "sef95", "aper_slope", "spec_ent"]
EV_SIG_FEATS = ["spin_n", "spin_dur", "spin_peak", "sw_n75", "sw_ptp"]
EV_EYE_FEATS = ["blink_n6", "blink_n4", "sem_n"]
B2_EMA_S = (30.0, 120.0)
B2_BASELINE_S = 120.0
EPS = 1e-6


def _idx(ch_names, name):
    names = list(ch_names)
    if name in names:
        return names.index(name)
    norm = [str(n).upper().replace("EEG", "").strip(" -_") for n in names]
    return norm.index(name.upper())


def bipolar(X, ch_names):
    """(B, C, T) -> (B, 6, T) derivations in BIP_PAIRS order."""
    return np.stack([X[:, _idx(ch_names, a)] - X[:, _idx(ch_names, b)] for a, b in BIP_PAIRS], axis=1)


def spectral12(S, sfreq):
    """(B, K, T) signals -> (B, K, 12) spectral features (BIP_FEATS order)."""
    S = S - S.mean(axis=-1, keepdims=True)
    freqs, psd = welch(S, fs=sfreq, nperseg=min(int(2 * sfreq), S.shape[-1]), axis=-1)
    df = freqs[1] - freqs[0]
    bp = {n: psd[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df for n, lo, hi in B2_BANDS}
    out = [np.log10(bp[n] + EPS) for n, _, _ in B2_BANDS]
    out.append(np.log10(bp["theta"] + EPS) - np.log10(bp["alpha"] + EPS))
    out.append(np.log10(bp["delta"] + bp["theta"] + EPS) - np.log10(bp["alpha"] + bp["beta"] + EPS))

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
    out += [peak(7, 13), peak(15, 30), sef, slope, ent]
    return np.stack(out, axis=-1)


_SOS = {}


def _sos(lo, hi, sfreq):
    key = (lo, hi, sfreq)
    if key not in _SOS:
        _SOS[key] = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    return _SOS[key]


def _runs(mask):
    """(N, T) bool -> list per row of (start, stop) index pairs of True runs."""
    d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    out = []
    for r in range(mask.shape[0]):
        s = np.flatnonzero(d[r] == 1)
        e = np.flatnonzero(d[r] == -1)
        out.append(list(zip(s, e)))
    return out


def events(S, sfreq):
    """(B, K, T) signals (uV) -> (B, K, 5) spindle / slow-wave features (EV_SIG_FEATS order)."""
    B, K, T = S.shape
    flat = (S - S.mean(axis=-1, keepdims=True)).reshape(B * K, T)
    sp = sosfiltfilt(_sos(11.0, 16.0, sfreq), flat, axis=-1)
    env = np.abs(hilbert(sp, axis=-1))
    thr = 2.5 * np.median(env, axis=-1, keepdims=True)
    runs = _runs(env > thr)
    lo_n, hi_n = int(0.5 * sfreq), int(2.0 * sfreq)
    spin = np.zeros((B * K, 3))
    for r, rr in enumerate(runs):
        ok = [(s, e) for s, e in rr if lo_n <= e - s <= hi_n]
        if ok:
            spin[r, 0] = len(ok)
            spin[r, 1] = sum(e - s for s, e in ok) / sfreq
            spin[r, 2] = max(env[r, s:e].max() for s, e in ok)
    spin[:, 2] = np.log10(spin[:, 2] + 1.0)
    sw = sosfiltfilt(_sos(0.5, 2.0, sfreq), flat, axis=-1)
    sgn = np.signbit(sw)
    zc = [np.flatnonzero(sgn[r, 1:] != sgn[r, :-1]) for r in range(B * K)]
    swf = np.zeros((B * K, 2))
    lo_w, hi_w = int(0.25 * sfreq), int(1.0 * sfreq)
    for r, z in enumerate(zc):
        best, n75 = 0.0, 0
        for a, b, c in zip(z[:-2], z[1:-1], z[2:]):  # half wave a..b, then b..c
            if not (lo_w <= b - a <= hi_w and lo_w <= c - b <= hi_w):
                continue
            ptp = sw[r, a:c].max() - sw[r, a:c].min()
            best = max(best, ptp)
            n75 += ptp > 75.0
        swf[r] = n75, np.log10(best + 1.0)
    return np.concatenate([spin, swf], axis=1).reshape(B, K, 5)


def eye_events(X, ch_names, sfreq):
    """(B, C, T) -> (B, 3) blink counts (two thresholds) and slow-eye-movement count (EV_EYE_FEATS order)."""
    i7, i8 = _idx(ch_names, "AF7"), _idx(ch_names, "AF8")
    front = 0.5 * (X[:, i7] + X[:, i8])
    front = sosfiltfilt(_sos(0.5, 10.0, sfreq), front - front.mean(axis=-1, keepdims=True), axis=-1)
    bip = X[:, i7] - X[:, i8]
    sem = sosfiltfilt(_sos(0.2, 0.6, sfreq), bip - bip.mean(axis=-1, keepdims=True), axis=-1)
    out = np.zeros((X.shape[0], 3))
    dist = max(1, int(0.3 * sfreq))
    for i in range(X.shape[0]):
        x = np.abs(front[i])
        mad = 1.4826 * np.median(np.abs(front[i] - np.median(front[i])))
        out[i, 0] = len(find_peaks(x, height=max(50.0, 6.0 * mad), distance=dist)[0])
        out[i, 1] = len(find_peaks(x, height=max(30.0, 4.0 * mad), distance=dist)[0])
        s = sem[i]
        z = np.flatnonzero(np.signbit(s[1:]) != np.signbit(s[:-1]))
        out[i, 2] = sum(np.abs(s[a:b]).max() > 20.0 for a, b in zip(z[:-1], z[1:]))
    return out


def window_features_b2(X, sfreq, ch_names, groups):
    """X: (B, C, T) microvolts -> (B, n) window-level B2 features for groups ("bip", "events")."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    out = []
    D = bipolar(X, ch_names)
    if "bip" in groups:
        out.append(spectral12(D, sfreq).reshape(len(X), -1))
    if "events" in groups:
        sig = np.concatenate([X, D], axis=1)
        out.append(events(sig, sfreq).reshape(len(X), -1))
        out.append(eye_events(X, ch_names, sfreq))
    return np.concatenate(out, axis=1) if out else np.zeros((len(X), 0))


def b2_window_names(ch_names, groups):
    bip = [f"{a}-{b}" for a, b in BIP_PAIRS]
    names = []
    if "bip" in groups:
        names += [f"{d}_{f}" for d in bip for f in BIP_FEATS]
    if "events" in groups:
        names += [f"{s}_{f}" for s in list(ch_names) + bip for f in EV_SIG_FEATS] + EV_EYE_FEATS
    return names


class B2Stream:
    """Causal stream features of the B2 groups for one recording; ``reset()`` before each recording.
    update(w, core) with w = window-level B2 vector (may be empty), core = (28,) core features of the window."""

    def __init__(self, step_s, epoch_group=False):
        self.step_s = float(step_s)
        self.alphas = [1.0 - np.exp(-self.step_s / t) for t in B2_EMA_S]
        self.per_epoch = max(1, int(round(30.0 / self.step_s)))
        self.epoch_group = epoch_group
        self.reset()

    def reset(self):
        self.n = 0
        self.emas = None
        self.base_sum = None
        self.base_n = 0
        self.cur_epoch = []   # core vectors of the epoch in progress
        self.epochs = []      # means of completed epochs (most recent last, at most 5 kept)

    def update(self, w, core):
        self.n += 1
        parts = []
        if w.size:
            if self.emas is None:
                self.emas = [w.copy() for _ in self.alphas]
                self.base_sum = np.zeros_like(w)
            else:
                for e, a in zip(self.emas, self.alphas):
                    e += a * (w - e)
            if self.n * self.step_s <= B2_BASELINE_S + 1e-9 or self.base_n == 0:
                self.base_sum += w
                self.base_n += 1
            base = self.base_sum / self.base_n
            cur = [w] + [e.copy() for e in self.emas]
            parts += cur + [c - base for c in cur]
        if self.epoch_group:
            phase = (self.n - 1) % self.per_epoch
            if self.epochs:
                last = self.epochs[-1]
                prev2 = np.mean(self.epochs[-3:-1], axis=0) if len(self.epochs) >= 2 else last
                prev4 = np.mean(self.epochs[-5:-1], axis=0) if len(self.epochs) >= 2 else last
                parts += [np.array([phase], float), last, prev2, prev4, last - prev4]
            else:  # no completed epoch yet: the running mean of the current one
                m = np.mean(self.cur_epoch + [core], axis=0)
                parts += [np.array([phase], float), m, m, m, np.zeros_like(m)]
            self.cur_epoch.append(core)
            if len(self.cur_epoch) == self.per_epoch:
                self.epochs.append(np.mean(self.cur_epoch, axis=0))
                self.epochs = self.epochs[-5:]
                self.cur_epoch = []
        return np.concatenate(parts) if parts else np.zeros(0)


def b2_stream_names(window_names, core_names, epoch_group):
    names = []
    if window_names:
        cur = list(window_names) + [f"{n}_ema{int(t)}s" for t in B2_EMA_S for n in window_names]
        names += cur + [f"{n}_rel" for n in cur]
    if epoch_group:
        names += ["epoch_phase"] + [f"{n}_{k}" for k in ("ep_last", "ep_prev2", "ep_prev4", "ep_trend") for n in core_names]
    return names
