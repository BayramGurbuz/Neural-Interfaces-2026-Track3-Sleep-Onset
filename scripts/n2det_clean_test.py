"""Input-cleaning test of a built submission whose bundle carries the N2 detector: corrupted windows must not reach
the detector (NaN / inf samples, a flat channel) and must not poison its causal summaries.

For 3 public-test recordings the wrapper is streamed window by window (deepcopy + reset_state, as the evaluator)
on (a) the clean signal, (b) a copy with injected faults: window 20 all NaN on channel 1, windows 40-42 with 30 %
random NaN on all channels, windows 60-70 channel 3 flat (constant), window 80 all channels +inf.
Checks: every prediction finite and within [0, 600]; every N2 summary finite; on the clean signal the summaries equal
the batched N2Summary of the detector's P (same code path as training) within 1e-5 (one window per call vs the whole
recording in one convolution batch differ by float32 rounding, ~5e-7); 5 min after the last fault the short-memory
summaries (P, EMA 30 s, 30 s rise) are back at the clean run's values (< 1e-3), i.e. the fault did not poison them.
EMA 2 min (tau 120 s) and the running maximum are reported only: the first still carries e^-2.5 of the fault, the second
is permanent by design. The prediction difference to the clean run is reported per fault region and 5 min after the last
fault; it does not vanish for any of our models (also without detector, mix3_540_gridpi): first-2-min baselines, running
extrema and the recurrent states keep a trace of what the cleaning rule substituted.

Usage: python scripts/n2det_clean_test.py SUBMISSION_DIR
"""
import copy
import importlib.util
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import torch

from ablation_c import ABLATIONS, build_table_ext
from train_hgb import SAVE

sub_dir = Path(sys.argv[1]).resolve()
stub = types.ModuleType("benchmark_utils.base_solver")
stub.CompetSolver = type("CompetSolver", (), {})
sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
sys.modules["benchmark_utils.base_solver"] = stub
spec = importlib.util.spec_from_file_location("submission", sub_dir / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
bundle = joblib.load(sub_dir / "model.joblib")
assert bundle.get("n2det_state") is not None or bundle.get("n2det_states") is not None, "bundle has no N2 detector"
wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, _, _ = build_table_ext(wg, sg)
zb = np.load(SAVE / "features_base.npz")
mt = zb["split"] == "test"
d = df[mt].reset_index(drop=True)
X = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
model = sub.HGBStreamModel(bundle.get("model"), 128.0, bundle.get("bin_edges"), bundle, [str(c) for c in zb["ch_names"]])
recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
recs = sorted(recs, key=len, reverse=True)[:3]  # the 3 longest test recordings
rng = np.random.default_rng(0)


def stream(x):
    m = copy.deepcopy(model)
    m.reset_state()
    preds, summ = [], []
    orig = m._n2det_rows

    def hook(xc):
        out = orig(xc)
        summ.append(out[0].copy())
        return out

    m._n2det_rows = hook
    for i in range(len(x)):
        preds.append(float(m.predict(torch.tensor(x[i][None]))[0]))
    return np.array(preds), np.stack(summ)


ok = True
for r in recs:
    x = X[r].copy()
    p0, s0 = stream(x)
    # batched reference of the summaries: detector P on the clean windows -> N2Summary
    with torch.no_grad():
        xr = torch.clamp(torch.tensor(x) / model.n2_scale, -model.n2_clip, model.n2_clip)
        if isinstance(model.n2det, list):  # multi-class detector: seed-mean closeness
            p = torch.stack([torch.softmax(n(xr)[0], -1) @ model.n2_w for n in model.n2det]).mean(0).numpy().astype(np.float64)
        else:
            p = torch.softmax(model.n2det(xr)[0], -1)[:, 1].numpy().astype(np.float64)
    st = sub.N2Summary(5.0)
    ref = np.stack([st.update(v) for v in p]).astype(np.float32)
    same = bool(np.abs(ref - s0).max() < 1e-5)
    xc = x.copy()
    faults = {"w20 ch1 NaN": [20], "w40-42 30% NaN": [40, 41, 42], "w60-70 ch3 flat": list(range(60, 71)), "w80 +inf": [80]}
    xc[20, 1] = np.nan
    for w in (40, 41, 42):
        xc[w][rng.random(xc[w].shape) < 0.3] = np.nan
    xc[60:71, 3] = 12.5
    xc[80] = np.inf
    p1, s1 = stream(xc)
    finite = np.isfinite(p1).all() and (p1 >= 0).all() and (p1 <= 600).all() and np.isfinite(s1).all()
    dlt = np.abs(p1 - p0)
    after = dlt[141:] if len(dlt) > 141 else dlt[-1:]  # 5 min after the last fault (window 80 + 60 windows)
    ds = np.abs(s1 - s0)[141:] if len(dlt) > 141 else np.zeros((1, 5))
    recovered = bool(ds[:, [0, 1, 4]].max() < 1e-3)
    ok &= same and finite and recovered
    print(f"recording {len(r)} windows: clean summaries == batched N2Summary (1e-5): {same}; corrupted run finite: {finite}; "
          f"P / EMA30 / rise30 recovered 5 min after the last fault: {recovered} (max |Δ| per summary "
          + " ".join(f"{v:.4f}" for v in ds.max(0)) + "); "
          f"|Δpred| corrupted vs clean: " + ", ".join(f"{k} {dlt[v].max():.1f} s" for k, v in faults.items())
          + f"; max over all windows {dlt.max():.1f} s; 5 min after the last fault max {after.max():.2f} s, mean {after.mean():.2f} s; "
          f"summaries |Δ| at the faults {np.abs(s1 - s0)[[20, 41, 65, 80]].max():.3f}")
print("PASS" if ok else "FAIL")
