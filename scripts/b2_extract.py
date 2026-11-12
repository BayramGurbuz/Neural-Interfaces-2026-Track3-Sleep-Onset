"""B2: window-level features of the "bip" and "events" groups for all 80 975 windows, same row order as
features_base.npz (raw windows from raw_windows.npy). Also times the per-window cost (single-window
calls, as in streaming inference).

Output: ~/neuralbench/save/features_b2.npz (feats_bip, feats_events, names_bip, names_events, timeline, stop, split)
Usage: python scripts/b2_extract.py
"""
import time

import numpy as np

from features_b2 import b2_window_names, window_features_b2
from train_hgb import SAVE

zb = np.load(SAVE / "features_base.npz")
ch = [str(c) for c in zb["ch_names"]]
raw = np.load(SAVE / "raw_windows.npy", mmap_mode="r")
print("channels", ch, "windows", len(raw), flush=True)

timing = {}
for g in ("bip", "events"):
    x = np.asarray(raw[:200])
    t0 = time.perf_counter()
    for i in range(len(x)):
        window_features_b2(x[i:i + 1], 128.0, ch, (g,))
    timing[g] = 1000 * (time.perf_counter() - t0) / len(x)
    print(f"group {g}: {timing[g]:.2f} ms per single window", flush=True)

out = {}
for g in ("bip", "events"):
    t0 = time.perf_counter()
    parts = [window_features_b2(np.asarray(raw[i:i + 2048]), 128.0, ch, (g,)) for i in range(0, len(raw), 2048)]
    out[f"feats_{g}"] = np.concatenate(parts).astype(np.float32)
    out[f"names_{g}"] = np.array(b2_window_names(ch, (g,)))
    f = out[f"feats_{g}"]
    print(f"{g}: {f.shape} in {time.perf_counter() - t0:.0f} s; NaN {int(np.isnan(f).sum())}, inf {int(np.isinf(f).sum())}", flush=True)
ev = out["feats_events"]
names = list(out["names_events"])
for k in ("blink_n6", "blink_n4", "sem_n", "TP9_spin_n", "TP10_spin_n", "TP9_sw_n75", "AF7-TP9_sw_n75"):
    j = names.index(k)
    print(f"  {k}: mean per window {ev[:, j].mean():.3f}, fraction > 0 {np.mean(ev[:, j] > 0):.3f}", flush=True)
np.savez(SAVE / "features_b2.npz", timeline=zb["timeline"], stop=zb["stop"], split=zb["split"],
         timing_ms=np.array([timing["bip"], timing["events"]]), **out)
print("saved", SAVE / "features_b2.npz")
