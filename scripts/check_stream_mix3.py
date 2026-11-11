"""Streaming-vs-batch equivalence and timing check of a built mix3 submission (no Docker).

Loads submissions/<name>/submission.py (benchmark_utils stubbed), streams every public-test recording
window by window exactly like the evaluator (fresh reset_state, batches of one (1, C, T) window), and
compares with the batched predictions saved by train_mix3_final.py (<save_dir>/test_preds.npz).

Usage: python scripts/check_stream_mix3.py NAME SAVE_DIR   e.g. mix3_540 ~/neuralbench/save/mix3_all
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from stream_utils import load_window_table, summarize
from train_hgb import SAVE

ROOT = Path(__file__).resolve().parents[1]
name, save_dir = sys.argv[1], Path(sys.argv[2]).expanduser()

stub = types.ModuleType("benchmark_utils.base_solver")
stub.CompetSolver = type("CompetSolver", (), {})
sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
sys.modules["benchmark_utils.base_solver"] = stub
spec = importlib.util.spec_from_file_location("submission", ROOT / "submissions" / name / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

bundle = joblib.load(ROOT / "submissions" / name / "model.joblib")
zb = np.load(SAVE / "features_base.npz")
ch_names = [str(c) for c in zb["ch_names"]]
mt = zb["split"] == "test"
raw = np.load(Path.home() / "neuralbench" / "save" / "raw_windows.npy", mmap_mode="r")
X_te = np.asarray(raw[mt])
meta = pd.DataFrame({"timeline": zb["timeline"][mt], "stop": zb["stop"][mt]})
ref = np.load(save_dir / "test_preds.npz", allow_pickle=True)
assert (ref["timeline"] == meta.timeline.to_numpy()).all() and (ref["stop"] == meta.stop.to_numpy()).all()
batched = ref["mix3_a0.7"]

model = sub.HGBStreamModel(bundle.get("model"), 128.0, bundle.get("bin_edges"), bundle, ch_names)
stream_pred = np.empty(len(meta))
t0 = time.perf_counter()
n_rec = 0
for tl, g in meta.groupby("timeline", sort=False):
    idx = g.sort_values("stop").index.to_numpy()
    model.reset_state()
    for i in idx:
        stream_pred[i] = float(model.predict(torch.tensor(X_te[i][None]))[0])
    n_rec += 1
dt = time.perf_counter() - t0
d_te = meta.merge(load_window_table(), on=["timeline", "stop"], how="left", validate="1:1")
diff = np.abs(stream_pred - batched)
print(f"{n_rec} recordings, {len(meta)} windows in {dt:.1f} s = {1000 * dt / len(meta):.2f} ms/window")
print(f"stream vs batched: max |Δ| {diff.max():.4f} s, mean {diff.mean():.5f} s, > 0.01 s in {(diff > 0.01).sum()} windows")
print(f"W-bMAE stream {summarize(d_te, stream_pred)['wbmae']:.4f}  batched {summarize(d_te, batched)['wbmae']:.4f}")
