"""Streaming-vs-batch equivalence and timing of a built mix3 submission with the 30 s grid decision
(or without: GRID_PI = 0), no Docker. Streams every public-test recording window by window like the
evaluator (deepcopy + reset_state per recording, batches of one window) and compares with the batched
decision of scripts/decision_layer.py computed from the same trained bundle.

Usage: python scripts/check_stream_grid.py SUBMISSION_DIR MODEL_DIR
  e.g. python scripts/check_stream_grid.py submissions/mix3_540_grid mix3_all
"""
import copy
import importlib.util
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import torch

from decision_layer import official_rec_scores, predict, test_fold
from train_hgb import SAVE

sub_dir, model_dir = Path(sys.argv[1]).resolve(), sys.argv[2]
stub = types.ModuleType("benchmark_utils.base_solver")
stub.CompetSolver = type("CompetSolver", (), {})
sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
sys.modules["benchmark_utils.base_solver"] = stub
spec = importlib.util.spec_from_file_location("submission", sub_dir / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
bundle = joblib.load(sub_dir / "model.joblib")
ft = test_fold(model_dir)
d = ft["dte"]
zb = np.load(SAVE / "features_base.npz")
mt = zb["split"] == "test"
assert (zb["timeline"][mt] == d.timeline.to_numpy()).all() and (zb["stop"][mt] == d.stop.to_numpy()).all()
X = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
batched = predict(ft, "old", sub.GRID_PI)
model = sub.HGBStreamModel(bundle.get("model"), 128.0, bundle.get("bin_edges"), bundle, [str(c) for c in zb["ch_names"]])
stream = np.empty(len(d))
t0 = time.perf_counter()
for _, g in d.groupby("timeline", sort=False):
    m = copy.deepcopy(model)
    m.reset_state()
    for i in g.sort_values("stop").index.to_numpy():
        stream[i] = float(m.predict(torch.tensor(X[i][None]))[0])
dt = time.perf_counter() - t0
diff = np.abs(stream - batched)
print(f"{sub_dir.name}: GRID_PI {sub.GRID_PI}; {d.timeline.nunique()} recordings, {len(d)} windows in {dt:.1f} s = "
      f"{1000 * dt / len(d):.2f} ms/window")
print(f"stream vs batched: max |Δ| {diff.max():.4f} s, mean {diff.mean():.5f} s, > 0.5 s in {(diff > 0.5).sum()} windows")
print(f"W-bMAE (official) stream {official_rec_scores(d, stream).mean():.4f}  batched {official_rec_scores(d, batched).mean():.4f}")
np.save(sub_dir.parent / f"{sub_dir.name}_stream_preds.npy", stream)
