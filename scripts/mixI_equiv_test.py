"""Work I equivalence test (19 Nov): the mixed package's stream predictions == the decision on 0.5 * W(ctx_c3_540) + 0.5 * W(ctxE_c2hs3p_540),
where W(pkg) is the pre-decision metric-weighted distribution each SUBMITTED package computes while streaming the 40 public-test recordings
(captured by wrapping the package module's `decide`). Also: each component package's own stream predictions are unchanged by the capture.

Usage: python scripts/mixI_equiv_test.py MIX_NAME PKG_A PKG_B [WEIGHT_A=0.5]   (names under submissions/)
       python scripts/mixI_equiv_test.py MIX_NAME PKG_A PKG_B PKG_C ... [--w=a,b,c]   (20 Nov, Work J: N packages, default equal weights)
"""
import importlib.util
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import torch

from train_hgb import SAVE

ROOT = Path(__file__).resolve().parents[1]
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
mix_name = pos[0]
try:  # old form: MIX A B WEIGHT_A
    wa = float(pos[3])
    pkgs, ws = pos[1:3], [wa, 1 - wa]
except (IndexError, ValueError):
    pkgs = pos[1:]
    ws = next(([float(x) for x in a.split("=", 1)[1].split(",")] for a in sys.argv if a.startswith("--w=")), [1 / len(pkgs)] * len(pkgs))
assert len(ws) == len(pkgs) >= 2
stub = types.ModuleType("benchmark_utils.base_solver")
stub.CompetSolver = type("CompetSolver", (), {})
sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
sys.modules["benchmark_utils.base_solver"] = stub
zb = np.load(SAVE / "features_base.npz")
mt = zb["split"] == "test"
X = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
tl = zb["timeline"][mt]
recs = [np.flatnonzero(tl == t) for t in dict.fromkeys(tl)]
ch = [str(c) for c in zb["ch_names"]]


def stream(name):
    spec = importlib.util.spec_from_file_location(f"sub_{name}", ROOT / "submissions" / name / "submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    cap = []
    orig = sub.decide

    def decide(W, *a, **k):
        cap.append(np.array(W, dtype=np.float64))
        return orig(W, *a, **k)

    sub.decide = decide
    b = joblib.load(ROOT / "submissions" / name / "model.joblib")
    m = sub.HGBStreamModel(b.get("model"), 128.0, b.get("bin_edges"), b, ch)
    pred = np.empty(len(X))
    for r in recs:
        m.reset_state()
        for i in r:
            pred[i] = float(m.predict(torch.tensor(X[i][None]))[0])
    W = np.concatenate(cap)
    return sub, orig, pred, W


_, _, p_mix, W_mix = stream(mix_name)
runs = [stream(p) for p in pkgs]
dec = runs[0][1]
for name, p in [(n, r[2]) for n, r in zip(pkgs, runs)] + [(mix_name, p_mix)]:
    ref = ROOT / "submissions" / f"{name}_stream_preds.npy"
    if ref.exists():
        print(f"{name}: stream predictions vs its saved check_stream run identical: {bool((np.load(ref) == p).all())}")
W = sum(w * r[3] for w, r in zip(ws, runs))
grid = np.asarray(joblib.load(ROOT / "submissions" / pkgs[0] / "model.joblib")["grid"], np.float64)
p_ref = np.clip(np.array([float(dec(W[i:i + 1], None, np.ones(W.shape[1]), grid)[0]) for i in range(len(W))]), 0, 600)
dW = np.abs(W_mix - W).max()
p_ref32 = p_ref.astype(np.float32).astype(np.float64)  # the wrapper returns float32 (resolution <= 6.1e-5 s below 600 s)
dp = np.abs(p_mix - p_ref32)
print(f"pre-decision distributions: max |W_mix - (" + " + ".join(f"{w:.4f} W_{n}" for w, n in zip(ws, pkgs)) + f")| = {dW:.2e}")
print(f"predictions vs float32-rounded reference: max |Δ| {dp.max():.2e} s, identical in {(dp == 0).mean():.4%} of windows, "
      f"> 0.01 s in {(dp > 0.01).sum()} windows")
print("PASS" if dW < 1e-9 and dp.max() <= 6.2e-5 else "FAIL")
