"""Work C wrapper tests (17 Nov, user): streaming context detector in a built submission.
(a) reset_state empties the context buffer and the detector summaries: stream recording A, reset, stream B -> B's closeness and
    predictions bit-identical to a fresh model on B; buffer empty and summaries reset right after reset_state.
(b) stream started at window k (k = 0, 1, 2, 3, 4, 5, 9; context < 6 for the first windows): per-window closeness of the wrapper
    == the training forward path on left-padded contexts from the same start (n2det_ctx.batched_closeness, CPU, 1e-5) and == the
    final training's crop-exact values (close_full_L + crop_closeness, GPU, 1e-4).
(c) NaN / flat channels: covered by extra_faults.py (n2_buf in the state check) and n2det_clean_test.py.

Usage: python scripts/ctx_wrapper_test.py SUBMISSION_DIR [DETECTOR_DIR=~/neuralbench/save/n2det_ctx_final]
"""
import copy
import importlib.util
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import torch

from n2det_ctx import batched_closeness
from n2det_feats import crop_closeness
from train_hgb import SAVE

sub_dir = Path(sys.argv[1]).resolve()
det_dir = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else SAVE / "n2det_ctx_final"
stub = types.ModuleType("benchmark_utils.base_solver")
stub.CompetSolver = type("CompetSolver", (), {})
sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
sys.modules["benchmark_utils.base_solver"] = stub
spec = importlib.util.spec_from_file_location("submission", sub_dir / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
LOG = []


class RecSummary(sub.N2Summary):  # records the per-window closeness the wrapper feeds into the summaries
    def update(self, p):
        LOG.append(float(p))
        return super().update(p)


sub.N2Summary = RecSummary
bundle = joblib.load(sub_dir / "model.joblib")
assert bundle.get("n2det_ctx_states") is not None, "not a context-detector bundle"
zb = np.load(SAVE / "features_base.npz")
mt = zb["split"] == "test"
tl = zb["timeline"][mt]
g_rows = np.flatnonzero(mt)
X = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
close_L = np.load(det_dir / "close_full_L.npy").astype(np.float64)
model = sub.HGBStreamModel(bundle.get("model"), 128.0, bundle.get("bin_edges"), bundle, [str(c) for c in zb["ch_names"]])
recs = [np.flatnonzero(tl == t) for t in dict.fromkeys(tl)]
recs = sorted(recs, key=len, reverse=True)


def stream(m, x):
    LOG.clear()
    preds = [float(m.predict(torch.tensor(x[i][None]))[0]) for i in range(len(x))]
    return np.array(preds), np.array(LOG)


ok = True
# (a) reset
A, B = X[recs[0][:80]], X[recs[1][:80]]
m = copy.deepcopy(model)
m.reset_state()
stream(m, A)
m.reset_state()
buf_empty = m.n2_buf == []
summ_fresh = m.n2_summ is None or (m.n2_summ.emas is None and m.n2_summ.runmax == -np.inf and m.n2_summ.hist == [])
pB, cB = stream(m, B)
f = copy.deepcopy(model)
f.reset_state()
pF, cF = stream(f, B)
same = bool(np.array_equal(pB, pF) and np.array_equal(cB, cF))
ok &= buf_empty and summ_fresh and same
print(f"(a) reset_state: context buffer empty {buf_empty}; detector summaries reset {summ_fresh}; "
      f"recording B after A + reset == fresh model: {same} (max |Δpred| {np.abs(pB - pF).max():.2e}, |Δclose| {np.abs(cB - cF).max():.2e})")

# (b) stream starts with context < 6
worst_cpu, worst_gpu = 0.0, 0.0
for r in recs[:3]:
    x = X[r]
    for k in (0, 1, 2, 3, 4, 5, 9):
        m = copy.deepcopy(model)
        m.reset_state()
        _, c = stream(m, x[k:k + 12])
        ref = batched_closeness(bundle["n2det_ctx_states"], x[:k + 12], model.n2_ctx, model.n2_scale, model.n2_clip, start=k)
        ref_gpu = crop_closeness(close_L, g_rows[r][k:k + 12])
        worst_cpu = max(worst_cpu, float(np.abs(c - ref).max()))
        worst_gpu = max(worst_gpu, float(np.abs(c - ref_gpu).max()))
        # sanity: the start-k values differ from the full-context ones in the first windows (the test is not vacuous)
        if k == 5 and r is recs[0]:
            full = crop_closeness(close_L, g_rows[r])[k:k + 12]
            print(f"    k = 5, first 5 windows: start-k closeness vs full-context closeness max |Δ| {np.abs(ref_gpu[:5] - full[:5]).max():.4f}")
pass_b = worst_cpu < 1e-5 and worst_gpu < 1e-4
ok &= pass_b
print(f"(b) streams from window k = 0, 1, 2, 3, 4, 5, 9 (3 recordings, 12 windows each): max |Δ| wrapper vs training forward (CPU) "
      f"{worst_cpu:.2e} (< 1e-5), vs final training crop-exact values (GPU) {worst_gpu:.2e} (< 1e-4): {pass_b}")
print("PASS" if ok else "FAIL")
