"""Streaming-vs-batch equivalence, component identity and timing of a built mix3 + N2-detector submission (no Docker).

Batched reference (training-time code paths, recomputed from the unbuilt bundle SAVE/<bundle>/model.joblib): sequence
components through gru_cv.ARCHS on padded recordings, CNN encoders through cnn_encoder.infer, HGB members through sklearn
predict_proba on the cached 817-feature table + N2Summary of the final detector's saved P (n2det_final/p_full.npy), mixed with
the bundle's alpha and decided without the grid. Component scores on the 40 public-test recordings (in-sample for 540
models) are printed next to the values logged by the training runs (--expect), which pins every part of the bundle to its
training run. Stream: submissions/<name>/submission.py, fresh reset_state per recording, one (1, C, T) window per call.

Usage: python scripts/check_stream_n2mix.py NAME [BUNDLE_DIR] [--expect=gru:37.4522,tcn:40.3806,gemb:49.5618,hgb:26.7506]
  without BUNDLE_DIR: streaming timing and score only (e.g. mix3_540 in the same session for the cost ratio).
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
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
name, bundle_dir = pos[0], (pos[1] if len(pos) > 1 else None)
expect = next((dict((kv.split(":")[0], float(kv.split(":")[1])) for kv in a.split("=", 1)[1].split(","))
               for a in sys.argv if a.startswith("--expect=")), {})

stub = types.ModuleType("benchmark_utils.base_solver")
stub.CompetSolver = type("CompetSolver", (), {})
sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
sys.modules["benchmark_utils.base_solver"] = stub
spec = importlib.util.spec_from_file_location("submission", ROOT / "submissions" / name / "submission.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

sb = joblib.load(ROOT / "submissions" / name / "model.joblib")
zb = np.load(SAVE / "features_base.npz")
ch_names = [str(c) for c in zb["ch_names"]]
mt = zb["split"] == "test"
X_te = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
meta = pd.DataFrame({"timeline": zb["timeline"][mt], "stop": zb["stop"][mt]})
d_te = meta.merge(load_window_table(), on=["timeline", "stop"], how="left", validate="1:1")

model = sub.HGBStreamModel(sb.get("model"), 128.0, sb.get("bin_edges"), sb, ch_names)
print(f"{name}: kind {sb['kind']}, alpha {sb.get('mix_alpha')}, n2det {'ctx' if sb.get('n2det_ctx_states') is not None else 'multi' if sb.get('n2det_states') is not None else 'binary' if sb.get('n2det_state') is not None else 'no'}"
      f" (into {tuple(sb.get('n2det_into', ()))}), GRID_ON {sub.GRID_ON}, groups {tuple(sb.get('groups', ()))}, "
      f"window groups {tuple(sb.get('window_groups', ()))}, {len(sb.get('feature_names', []))} HGB features")
stream_pred = np.empty(len(meta))
t0 = time.perf_counter()
n_rec = 0
for tl, g in meta.groupby("timeline", sort=False):
    model.reset_state()
    for i in g.sort_values("stop").index.to_numpy():
        stream_pred[i] = float(model.predict(torch.tensor(X_te[i][None]))[0])
    n_rec += 1
dt = time.perf_counter() - t0
print(f"{n_rec} recordings, {len(meta)} windows in {dt:.1f} s = {1000 * dt / len(meta):.2f} ms/window; "
      f"stream W-bMAE {summarize(d_te, stream_pred)['wbmae']:.4f}; all finite {bool(np.isfinite(stream_pred).all())}")
np.save(ROOT / "submissions" / f"{name}_stream_preds.npy", stream_pred)
if bundle_dir is None:
    sys.exit(0)

# ---- batched reference from the unbuilt bundle
from ablation_c import ABLATIONS, build_table_ext  # noqa: E402
from cnn_encoder import Encoder, infer  # noqa: E402
from cv_utils import apply_bin_edges  # noqa: E402
from dist_hgb import decide  # noqa: E402
from ensemble_dist import GRID, to_grid  # noqa: E402
from gru_cv import ARCHS, CONFIGS, DEV, pad, seq_inputs  # noqa: E402
from n2det_stream import N2Summary  # noqa: E402
from n2det_feats import summaries  # noqa: E402

b = joblib.load(Path(bundle_dir).expanduser() / "model.joblib")
wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, X, _ = build_table_ext(wg, sg)
assert tuple(b["groups"]) == tuple(sg) and tuple(b["window_groups"]) == tuple(wg)
dmt = (df.split == "test").to_numpy()
dd = df[dmt].reset_index(drop=True)
assert (dd.timeline.to_numpy() == meta.timeline.to_numpy()).all() and (dd.stop.to_numpy() == meta.stop.to_numpy()).all()
ze = np.load(SAVE / "features_ext.npz")
Fte = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)[mt]
te_recs = [g.sort_values("stop").index.to_numpy() for _, g in dd.groupby("timeline", sort=False)]
B, omega_b = np.asarray(b["gru_edges"], np.float64), np.asarray(b["gru_omega_b"], np.float64)


def seq_probs(arch, states, mu, sd, cfg, extra=None):
    out = []
    for sdict in states:
        net = ARCHS[arch](len(mu), len(B) - 1, cfg["hidden"], cfg["dropout"], cfg["in_dropout"]).to(DEV)
        net.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
        net.eval()
        xt = torch.tensor(pad([(seq_inputs(Fte[r], None if extra is None else extra[r]) - mu) / sd for r in te_recs]), device=DEV)
        with torch.no_grad():
            P = torch.softmax(net(xt)[0], -1).cpu().numpy()
        probs = np.empty((len(dd), len(B) - 1), np.float32)
        for i, r in enumerate(te_recs):
            probs[r] = P[i, :len(r)]
        out.append(probs)
    return np.mean(out, axis=0).astype(np.float64)


def weighted_grid(P):
    q = P * omega_b
    return to_grid(q / q.sum(1, keepdims=True), B)


n2v = None  # per-window detector output for all 540 rows (final model, as the wrapper computes it)
if b.get("n2det_ctx_states") is not None:  # context detector, context from the recording start (= L = 6 row)
    n2v = np.load(SAVE / "n2det_ctx_final" / "close_full.npy").astype(np.float64)
elif b.get("n2det_states") is not None:
    n2v = np.load(SAVE / "n2det_final" / "close_full.npy").astype(np.float64)
elif b.get("n2det_state") is not None:
    n2v = np.load(SAVE / "n2det_final" / "p_full.npy").astype(np.float64)
if n2v is not None and b.get("n2det_ctx_states") is not None and "--cpu-det" in sys.argv:
    # 18 Nov: the context detector's test-row outputs recomputed on the CPU with the training forward path (n2det_ctx.batched_closeness)
    # instead of the GPU-computed close_full (differences ~3e-5 can flip an HGB split right at a bin edge)
    from n2det_ctx import batched_closeness  # noqa: E402
    n2v = n2v.copy()
    g_te = np.flatnonzero(mt)
    for r in te_recs:
        n2v[g_te[r]] = batched_closeness(b["n2det_ctx_states"], X_te[r], int(b["n2det_ctx_n"]), float(b["n2det_scale"]), float(b["n2det_clip"]))
    print("context detector reference: CPU training forward path", flush=True)
n2_te = None if n2v is None else summaries(n2v[mt], te_recs)  # (test rows, 5), per recording from its first window
Pg = seq_probs("gru", b["gru_states"], b["gru_mu"], b["gru_sd"], CONFIGS["uw"], extra=n2_te if "gru" in tuple(b.get("n2det_into", ())) else None)
comps = tuple(b.get("seq_comps", ("gru", "tcn", "gemb")))  # 17 Nov: subsets (candidate 3: GRU only)
Pt = seq_probs("tcn", b["tcn_states"], b["tcn_mu"], b["tcn_sd"], CONFIGS["tcn_n10"]) if "tcn" in comps else None
Xr = torch.as_tensor(X_te, device=DEV)
Pe = []
for sdict, st in (zip(b["cnn_states"], b["gemb_states"]) if "gemb" in comps else ()):
    enc = Encoder().to(DEV)
    enc.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
    enc.eval()
    _, e = infer(enc, Xr, torch.arange(len(dd), device=DEV))
    Pe.append(seq_probs("gru", [st], b["gemb_mu"], b["gemb_sd"], CONFIGS["uw"], extra=e.cpu().numpy().astype(np.float32)))
Pe = np.mean(Pe, axis=0) if Pe else None
Xt = X[dmt]
if n2v is not None:
    assert set(b.get("n2det_into", ())) <= {"gru"}, "TCN / GRU-emb with detector inputs are not covered by this check"
    n2 = np.zeros((len(df), 5), np.float32)
    st = N2Summary(5.0)
    for _, g in df.groupby("timeline", sort=False):
        st.reset()
        for i in g.sort_values("stop").index.to_numpy():
            n2[i] = st.update(n2v[i])
    assert np.abs(n2[dmt] - n2_te).max() < 1e-6
    Xt = np.concatenate([Xt, n2[dmt]], axis=1)
Xb = apply_bin_edges(Xt, b["bin_edges"])
Wh = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(b["models"], b["member_edges"])], axis=0) * b["grid_omega"]
Wh /= Wh.sum(1, keepdims=True)
alpha = float(b["mix_alpha"])
ones = np.ones(len(GRID) - 1)
qs = {k: weighted_grid(P) for k, P in (("gru", Pg), ("tcn", Pt), ("gemb", Pe)) if k in comps}
cands = {**qs, "hgb": Wh, "mix": alpha * sum(qs[k] for k in comps) / len(comps) + (1 - alpha) * Wh}
ok = True
for k, W in cands.items():
    s = summarize(dd, decide(W, None, ones, True, GRID))["wbmae"]
    note = ""
    if k in expect:
        match = abs(s - expect[k]) < 5e-4
        ok &= match
        note = f"  (training log {expect[k]:.4f}: {'match' if match else 'MISMATCH'})"
    print(f"batched {k:5s} {s:.4f}{note}")
batched = decide(cands["mix"], None, ones, True, GRID)
diff = np.abs(stream_pred - batched)
print(f"stream vs batched (alpha {alpha}): max |Δ| {diff.max():.4f} s, mean {diff.mean():.5f} s, > 0.01 s in {(diff > 0.01).sum()} "
      f"windows, > 0.5 s in {(diff > 0.5).sum()}; W-bMAE stream {summarize(dd, stream_pred)['wbmae']:.4f} "
      f"batched {summarize(dd, batched)['wbmae']:.4f}")
ok &= bool(diff.max() < 0.5)
print("PASS" if ok else "FAIL")
