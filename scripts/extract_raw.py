"""Cache the raw 5 s windows (N, 4, 640) float32 microvolts for all splits, in the same row order
as features_base.npz (asserted on timeline / stop), for the CNN window encoder (cnn_encoder.py).

Output: ~/neuralbench/save/raw_windows.npy (~830 MB).
Usage: python scripts/extract_raw.py
"""
import time
from pathlib import Path

import numpy as np
from neuralbench.experiment_config import build_experiment_configs
from neuralbench.main import Experiment
from torch.utils.data import DataLoader

OUT = Path.home() / "neuralbench" / "save" / "raw_windows.npy"
BASE = Path.home() / "neuralbench" / "save" / "features_base.npz"

cfg = build_experiment_configs("eeg", "_sleep_onset_stream", model="eegnet", quiet=True)[0]
exp = Experiment(**cfg)
loaders = exp.data.prepare()

raw, timelines, stops, splits = [], [], [], []
for split in ("train", "val", "test"):
    t0 = time.perf_counter()
    ds = loaders[split].dataset
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=ds.collate_fn, num_workers=8)
    for batch in dl:
        raw.append(batch.data["neuro"].numpy().astype(np.float32))
        timelines += [s.timeline for s in batch.segments]
        stops += [s.stop for s in batch.segments]
        splits += [split] * len(raw[-1])
    print(f"{split}: {len(ds)} windows in {time.perf_counter() - t0:.0f} s", flush=True)

X = np.concatenate(raw)
zb = np.load(BASE)
assert (zb["timeline"] == np.array(timelines)).all() and (zb["stop"] == np.array(stops)).all() \
    and (zb["split"] == np.array(splits)).all(), "row order differs from features_base.npz"
print("shape", X.shape, "median |x|", float(np.median(np.abs(X))), "p99.9 |x|", float(np.percentile(np.abs(X), 99.9)))
np.save(OUT, X)
print("saved", OUT, f"{X.nbytes / 1e6:.0f} MB")
