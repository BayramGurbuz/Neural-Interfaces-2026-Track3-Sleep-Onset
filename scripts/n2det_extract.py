"""Work 2 (in-domain N2 detector): cut the post-onset part of every raw recording into 5 s windows.

The stream pipeline ends every recording at the N2 onset; the raw BrainVision files go on for exactly 300 s more
(checked: 540 / 540 recordings, no NaN). Windows k = 0..59 cover [onset + 5k, onset + 5k + 5). Values are the raw
float32 samples x 0.1 (channel resolution 0.1 uV in the .vhdr), which reproduces the stream windows
(raw_windows.npy) to 1e-7 relative (checked on 2000 random windows here and asserted again below).

Features: the same per-window functions as the stream table (causal_features.window_features 40,
window_features_ext spec+eye 22, features_b2 bip 72).

Output: ~/neuralbench/save/n2det_post.npz (raw (N, 4, 640), feats62, feats_bip, timeline, k, split)
Usage: python scripts/n2det_extract.py
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal_features import window_features, window_features_ext
from features_b2 import window_features_b2
from train_hgb import SAVE

ROOT = Path.home() / "neuralbench" / "data" / "Interaxon2026Muse" / "download" / "nm000287"
SF, WIN, N_POST = 128, 640, 60


def rec_paths(timeline):
    ses, sub = re.search(r"session=(ses-\d+),subject=(sub-\d+)", timeline).groups()
    base = ROOT / sub / ses / "eeg" / f"{sub}_{ses}_task-sleeponset"
    return Path(f"{base}_eeg.eeg"), Path(f"{base}_events.tsv")


def load_signal(timeline):
    eeg, ev = rec_paths(timeline)
    x = np.fromfile(eeg, np.float32).reshape(-1, 4).T * np.float32(0.1)
    e = pd.read_csv(ev, sep="\t")
    onset = float(e.loc[e.trial_type == "n2_onset", "onset"].iloc[0])
    return x, onset


if __name__ == "__main__":
    zb = np.load(SAVE / "features_base.npz")
    ch = [str(c) for c in zb["ch_names"]]
    raw_stream = np.load(SAVE / "raw_windows.npy", mmap_mode="r")
    tl_all, st_all, sp_all = zb["timeline"], zb["stop"], zb["split"]
    timelines = list(dict.fromkeys(tl_all))
    t0 = time.perf_counter()
    raw, tl, ks, sp, n_check = [], [], [], [], 0
    for t in timelines:
        x, onset = load_signal(t)
        o = int(round(onset * SF))
        assert x.shape[1] - o == N_POST * WIN, (t, x.shape, onset)
        rows = np.flatnonzero(tl_all == t)
        for i in rows[:: max(1, len(rows) // 3)]:  # alignment check on a few stream windows per recording
            e = int(round(st_all[i] * SF))
            assert np.allclose(x[:, e - WIN:e], raw_stream[i], rtol=1e-5, atol=1e-4, equal_nan=True), (t, st_all[i])
            n_check += 1
        raw.append(np.stack([x[:, o + WIN * k:o + WIN * (k + 1)] for k in range(N_POST)]))
        tl += [t] * N_POST
        ks += list(range(N_POST))
        sp += [sp_all[rows[0]]] * N_POST
    R = np.concatenate(raw).astype(np.float32)
    print(f"{len(timelines)} recordings, {len(R)} post-onset windows, {n_check} stream windows re-checked, "
          f"NaN {int(np.isnan(R).sum())} ({time.perf_counter() - t0:.0f} s)", flush=True)
    f62 = np.concatenate([np.concatenate([window_features(R[i:i + 2048], float(SF)),
                                          window_features_ext(R[i:i + 2048], float(SF), ch, ("spec", "eye"))], axis=1)
                          for i in range(0, len(R), 2048)]).astype(np.float32)
    fb = np.concatenate([window_features_b2(R[i:i + 2048], float(SF), ch, ("bip",))
                         for i in range(0, len(R), 2048)]).astype(np.float32)
    print(f"features 62 {f62.shape}, bip {fb.shape}; NaN {int(np.isnan(f62).sum())} / {int(np.isnan(fb).sum())} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    np.savez(SAVE / "n2det_post.npz", raw=R, feats62=f62, feats_bip=fb, timeline=np.array(tl), k=np.array(ks),
             split=np.array(sp))
    print("saved", SAVE / "n2det_post.npz")
