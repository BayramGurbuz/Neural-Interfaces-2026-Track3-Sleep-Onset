"""Stream-start robustness of the ungridded mix3 recipe, no retraining (14 Nov: sealed streams may not start at the
recording start). The train+val mix3 bundle (SAVE/mix3: GRU-uw x10, TCN x10, CNN enc + GRU-emb x3, ensemble (c);
all trained on the 500 train+val recordings) is run on the 40 public-test recordings (out-of-sample; seen subjects only,
so no unseen / seen split) with the first k windows of every recording dropped and the stream restarted there:
elapsed time, EMAs / dynamics / running extrema of the 817 HGB features, the first-2-min baselines and the GRU / TCN
states all start at window k. Targets are unchanged. k = 0, 6 (30 s), 24 (2 min), 60 (5 min) and random per
recording (uniform integer in [0, (onset - 60 s) / 5], seed 0).

Scores (official per-recording W-bMAE, mean over recordings, with the 4 bin contributions) over the windows >= k:
  (a) truncated stream (model history starts at k), (b) the full-history predictions (k = 0 run) on the same windows.
  Δ = (a) - (b) isolates the model's degradation from the change of scored windows.
Checks: the re-streamed 817 features at k = 0 reproduce the cached table; k = 0 reproduces the saved mix3 test score.

Usage: python scripts/stream_start_test.py   -> OUT/stream_start_test.json
"""
import json
import time

import joblib
import numpy as np
import torch

from ablation_c import ABLATIONS, build_table_ext
from causal_features import CausalFeatureStream, core_index
from cnn_encoder import Encoder, infer
from cv_utils import apply_bin_edges
from dist_hgb import decide
from dual_cv import OUT
from ensemble_dist import GRID, to_grid
from gru_cv import ARCHS, CONFIGS, seq_inputs
from train_hgb import SAVE

KS = {"k0": 0, "k6_30s": 6, "k24_2min": 24, "k60_5min": 60, "random": None}
BIN_EDGES, BIN_W = np.array([0.0, 40.0, 90.0, 300.0, 600.1]), np.array([10.0, 5.0, 3.0, 1.0])
BIN_NAMES = ["[0,40)", "[40,90)", "[90,300)", "[300,600]"]

b = joblib.load(SAVE / "mix3" / "model.joblib")
wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, X817, _ = build_table_ext(wg, sg)
mt = (df.split == "test").to_numpy()
d = df[mt].reset_index(drop=True)
X817 = X817[mt]
zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
F = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)[mt]
raw = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
target = d.target.to_numpy().astype(np.float64)
onset = {i: float(d.stop[r[-1]] + target[r[-1]]) for i, r in enumerate(recs)}
B, omega_b, alpha = np.asarray(b["gru_edges"], np.float64), np.asarray(b["gru_omega_b"], np.float64), float(b["mix_alpha"])
cfg = CONFIGS["uw"]
stream = CausalFeatureStream(5.0, tuple(b["groups"]), core_index(range(4)))


def nets(arch, states, n_in):
    out = []
    for s in states:
        net = ARCHS[arch](n_in, len(B) - 1, cfg["hidden"], cfg["dropout"], cfg["in_dropout"])
        net.load_state_dict({k: torch.tensor(v) for k, v in s.items()})
        out.append(net.eval())
    return out


grus, tcns, gembs = nets("gru", b["gru_states"], len(b["gru_mu"])), nets("tcn", b["tcn_states"], len(b["tcn_mu"])), \
    nets("gru", b["gemb_states"], len(b["gemb_mu"]))
encs = []
for s in b["cnn_states"]:
    e = Encoder()
    e.load_state_dict({k: torch.tensor(v) for k, v in s.items()})
    encs.append(e.eval())


def seq_mean(models, x):
    with torch.no_grad():
        return np.mean([torch.softmax(m(torch.tensor(x)[None])[0][0], -1).numpy().astype(np.float64) for m in models], axis=0)


def wgrid(P):
    q = P * omega_b
    return to_grid(q / q.sum(1, keepdims=True), B)


def predict_from(r, k):
    """Predictions for the windows r[k:] of one recording with the stream restarted at window k."""
    rows = r[k:]
    Fw = F[rows]
    stream.reset()
    H = np.stack([stream.update(fi.astype(np.float64)) for fi in Fw]).astype(np.float32)
    Xb = apply_bin_edges(H, b["bin_edges"])
    Ph = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(b["models"], b["member_edges"])], axis=0)
    Wh = Ph * np.asarray(b["grid_omega"])
    Wh /= Wh.sum(1, keepdims=True)
    with torch.no_grad():
        embs = [infer(e, torch.tensor(raw[rows]), torch.arange(len(rows)))[1].numpy().astype(np.float32) for e in encs]
    xg = seq_inputs(Fw)
    qg = wgrid(seq_mean(grus, (xg - b["gru_mu"]) / b["gru_sd"]))
    qt = wgrid(seq_mean(tcns, (xg - b["tcn_mu"]) / b["tcn_sd"]))
    qe = wgrid(np.mean([seq_mean([g], (seq_inputs(Fw, E) - b["gemb_mu"]) / b["gemb_sd"]) for g, E in zip(gembs, embs)], axis=0))
    W = alpha * (qg + qt + qe) / 3 + (1 - alpha) * Wh
    return decide(W, None, np.ones(W.shape[1]), True, GRID), H


def score(rows_list, preds_list):
    """Official W-bMAE per recording (mean over recordings) and its bin contributions."""
    tot, contrib = [], []
    for rows, p in zip(rows_list, preds_list):
        y = target[rows]
        w = BIN_W[np.searchsorted(BIN_EDGES, y, side="right") - 1]
        e = w * np.abs(p - y)
        c = np.array([e[(y >= lo) & (y < hi)].sum() for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]) / w.sum()
        tot.append(e.sum() / w.sum())
        contrib.append(c)
    return float(np.mean(tot)), np.mean(contrib, axis=0).tolist()


if __name__ == "__main__":
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    full, Hs = [], []
    for r in recs:
        p, H = predict_from(r, 0)
        full.append(p)
        Hs.append(H)
    dev = max(float(np.abs(H - X817[r]).max()) for H, r in zip(Hs, recs))
    s0, c0 = score(recs, full)
    print(f"k = 0: re-streamed 817 features vs cached table max |Δ| {dev:.2e}; W-bMAE {s0:.4f} (saved mix3 train+val test "
          f"score 47.0597); {time.perf_counter() - t0:.0f} s", flush=True)
    res = {"n_recordings": len(recs), "k0_wbmae": s0, "k0_contrib": c0, "feature_check_max_abs": dev, "runs": {}}
    for name, k in KS.items():
        ks = [k if k is not None else int(rng.integers(0, max(int((onset[i] - 60) // 5), 0) + 1)) for i in range(len(recs))]
        ks = [min(kk, len(r) - 1) for kk, r in zip(ks, recs)]
        rows_list = [r[kk:] for r, kk in zip(recs, ks)]
        trunc = [predict_from(r, kk)[0] for r, kk in zip(recs, ks)]
        ref = [p[kk:] for p, kk in zip(full, ks)]
        st, ct = score(rows_list, trunc)
        sr, cr = score(rows_list, ref)
        res["runs"][name] = dict(k=ks if k is None else k, n_windows=int(sum(len(x) for x in rows_list)),
                                 truncated=st, full_history=sr, delta=st - sr, contrib_truncated=ct, contrib_full=cr,
                                 d_contrib=(np.array(ct) - np.array(cr)).tolist())
        print(f"{name:9s} windows {res['runs'][name]['n_windows']:5d} | truncated {st:6.2f}  full-history {sr:6.2f}  "
              f"Δ {st - sr:+6.2f} | Δ contrib " + " ".join(f"{n} {x:+.2f}" for n, x in zip(BIN_NAMES, res["runs"][name]["d_contrib"]))
              + (f" | k median {int(np.median(ks))}, max {max(ks)}" if k is None else ""), flush=True)
    res["time_s"] = time.perf_counter() - t0
    (OUT / "stream_start_test.json").write_text(json.dumps(res, indent=2))
    print(f"saved {OUT / 'stream_start_test.json'} ({res['time_s']:.0f} s)")
