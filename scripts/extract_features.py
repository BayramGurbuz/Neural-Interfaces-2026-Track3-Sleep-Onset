"""Compute per-window base features for all splits once and cache them.

Windows come from NeuralBench's ``_sleep_onset_stream`` pipeline (raw microvolts, no filter /
scaler: the same signal the Codabench evaluator feeds to ``predict``). Output:
~/neuralbench/save/features_base.npz with feats (N, 40), timeline, stop, split, ch_names.

Usage: python scripts/extract_features.py
"""
import time
from pathlib import Path

import numpy as np
from neuralbench.experiment_config import build_experiment_configs
from neuralbench.main import Experiment
from torch.utils.data import DataLoader

from causal_features import window_features

OUT = Path.home() / "neuralbench" / "save" / "features_base.npz"

cfg = build_experiment_configs("eeg", "_sleep_onset_stream", model="eegnet", quiet=True)[0]
exp = Experiment(**cfg)
loaders = exp.data.prepare()
ds0 = loaders["train"].dataset
chans = getattr(ds0.extractors["neuro"], "_channels", None)
ch_names = [n for n, _ in sorted(chans.items(), key=lambda kv: kv[1])] if chans else None
print("channels:", ch_names, flush=True)
sfreq = 128.0

feats, timelines, stops, splits = [], [], [], []
for split in ("train", "val", "test"):
    t0 = time.perf_counter()
    ds = loaders[split].dataset
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=ds.collate_fn, num_workers=8)
    for i, batch in enumerate(dl):
        X = batch.data["neuro"].numpy()
        if i == 0:
            print(split, "X", X.shape, X.dtype, "median |x|", float(np.median(np.abs(X))),
                  "channel means", X.mean(axis=(0, 2)).round(1), flush=True)
        feats.append(window_features(X, sfreq))
        timelines += [s.timeline for s in batch.segments]
        stops += [s.stop for s in batch.segments]
        splits += [split] * len(X)
    print(f"{split}: {len(ds)} windows in {time.perf_counter() - t0:.0f} s", flush=True)

np.savez(OUT, feats=np.concatenate(feats).astype(np.float32), timeline=np.array(timelines),
         stop=np.array(stops), split=np.array(splits), ch_names=np.array(ch_names))
print("saved", OUT)
