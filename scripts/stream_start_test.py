"""Stream-start robustness of the ungridded mix3 recipe, no retraining (14 Nov: sealed streams may not start at the
recording start). The train+val mix3 bundle (SAVE/mix3: GRU-uw x10, TCN x10, CNN enc + GRU-emb x3, ensemble (c);
all trained on the 500 train+val recordings) is run on the 40 public-test recordings (out-of-sample; seen subjects only,
so no unseen / seen split) with the first k windows of every recording dropped and the stream restarted there:
elapsed time, EMAs / dynamics / running extrema of the 817 HGB features, the first-2-min baselines and the GRU / TCN
states all start at window k. Targets are unchanged. k = 0, 6 (30 s), 24 (2 min), 60 (5 min) and random per
recording (uniform integer in [0, (onset - 60 s) / 5], seed 0).

Scores (official per-recording W-bMAE = bin-weighted mean of per-bin MAEs, mean over recordings; 4 bin contributions) over the windows >= k:
  (a) truncated stream (model history starts at k), (b) the full-history predictions (k = 0 run) on the same windows.
  Δ = (a) - (b) isolates the model's degradation from the change of scored windows.
Checks: the re-streamed 817 features at k = 0 reproduce the cached table; k = 0 reproduces the saved mix3 test score.

Usage: python scripts/stream_start_test.py [--bundle=mix3] [--gru-from=DIR] [--tag=T]
  --bundle: SAVE/<dir>/model.joblib to evaluate (default mix3 = train+val, no augmentation); --gru-from: take only the
  GRU-uw states (and their mu / sd) from another bundle (e.g. the crop-augmented one); -> OUT/stream_start_test<T>.json
"""
import json
import sys
import time

import joblib
import numpy as np
import torch

from ablation_c import ABLATIONS, build_table_ext
from causal_features import CausalFeatureStream, core_index
from cnn_encoder import Encoder, infer
from cv_utils import apply_bin_edges
from dist_hgb import decide
from decision_layer import RANGE_NAMES, WEIGHTS, official_rec_scores, per_recording_scores
from dual_cv import OUT
from ensemble_dist import GRID, to_grid
from gru_cv import ARCHS, CONFIGS, seq_inputs
from train_hgb import SAVE

KS = {"k0": 0, "k1_5s": 1, "k3_15s": 3, "k6_30s": 6, "k12_60s": 12, "k24_2min": 24, "k60_5min": 60, "random": None}
ARG = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
BUNDLE, GRU_FROM, TAG = ARG.get("--bundle", "mix3"), ARG.get("--gru-from"), ARG.get("--tag", "")
BIN_NAMES = ["[0,40)", "[40,90)", "[90,300)", "[300,600]"]

b = joblib.load(SAVE / BUNDLE / "model.joblib")
if GRU_FROM:
    bg = joblib.load(SAVE / GRU_FROM / "model.joblib")
    b.update(gru_states=bg["gru_states"], gru_mu=bg["gru_mu"], gru_sd=bg["gru_sd"])
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


EL_H = list(build_table_ext(wg, sg)[2]).index("elapsed_s")  # elapsed column of the 817 HGB features
EMB_N = int(b["cnn_emb"])  # encoder embedding width (GRU-emb extras)
EL_S = 2 * F.shape[1]  # log1p-elapsed column of the sequence inputs (seq_inputs: [F, F - base, el, extras...])


def predict_from(r, k, true_elapsed=False):
    """Predictions for the windows r[k:] of one recording with the stream restarted at window k -> dict of
    mix / hgb-only / seq-only decisions and the re-streamed HGB features. true_elapsed: the elapsed-time inputs
    (HGB elapsed_s, sequence log1p elapsed) are replaced by the elapsed time since the recording start (diagnostic:
    isolates the elapsed input from the loss of EMAs / baselines / recurrent states)."""
    rows = r[k:]
    Fw = F[rows]
    stream.reset()
    H = np.stack([stream.update(fi.astype(np.float64)) for fi in Fw]).astype(np.float32)
    xg = seq_inputs(Fw)
    if true_elapsed:
        H[:, EL_H] = (k + np.arange(1, len(rows) + 1)) * 5.0
        xg[:, EL_S] = np.log1p(np.minimum((k + np.arange(1, len(rows) + 1)) * 5.0, 1500.0))
    Xb = apply_bin_edges(H, b["bin_edges"])
    Ph = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(b["models"], b["member_edges"])], axis=0)
    Wh = Ph * np.asarray(b["grid_omega"])
    Wh /= Wh.sum(1, keepdims=True)
    with torch.no_grad():
        embs = [infer(e, torch.tensor(raw[rows]), torch.arange(len(rows)))[1].numpy().astype(np.float32) for e in encs]
    qg = wgrid(seq_mean(grus, (xg - b["gru_mu"]) / b["gru_sd"]))
    qt = wgrid(seq_mean(tcns, (xg - b["tcn_mu"]) / b["tcn_sd"]))
    xe = [seq_inputs(Fw, E) for E in embs]
    if true_elapsed:
        for x in xe:
            x[:, EL_S] = xg[:, EL_S]
    qe = wgrid(np.mean([seq_mean([g], (x - b["gemb_mu"]) / b["gemb_sd"]) for g, x in zip(gembs, xe)], axis=0))
    qs = (qg + qt + qe) / 3
    ones = np.ones(Wh.shape[1])
    return dict(mix=decide(alpha * qs + (1 - alpha) * Wh, None, ones, True, GRID), hgb=decide(Wh, None, ones, True, GRID),
                seq=decide(qs, None, ones, True, GRID), H=H)


def score(rows_list, preds_list):
    """Official per-recording W-bMAE (decision_layer.official_rec_scores: MAE per bin, bin-weighted mean over the bins
    present in the recording), mean over recordings; contributions = weighted per-bin MAE / sum of present weights."""
    rows = np.concatenate(rows_list)
    dte = d.iloc[rows].reset_index(drop=True)
    pred = np.concatenate(preds_list)
    tot = float(official_rec_scores(dte, pred).mean())
    local = per_recording_scores(dte, pred)
    mae = local[RANGE_NAMES].to_numpy()
    w = np.where(np.isnan(mae), 0.0, WEIGHTS)
    contrib = (np.nan_to_num(mae) * w / w.sum(1, keepdims=True)).mean(0)
    assert abs(contrib.sum() - tot) < 1e-6, (contrib.sum(), tot)
    return tot, contrib.tolist()


if __name__ == "__main__":
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    full = [predict_from(r, 0) for r in recs]
    dev = max(float(np.abs(f["H"] - X817[r]).max()) for f, r in zip(full, recs))
    s0, c0 = score(recs, [f["mix"] for f in full])
    print(f"bundle {BUNDLE}" + (f" + GRU from {GRU_FROM}" if GRU_FROM else "") + f"; k = 0: re-streamed 817 features vs cached table max |Δ| {dev:.2e}; W-bMAE mix {s0:.4f} (saved mix3 train+val "
          f"test score 47.0597), hgb-only {score(recs, [f['hgb'] for f in full])[0]:.2f}, seq-only "
          f"{score(recs, [f['seq'] for f in full])[0]:.2f}; {time.perf_counter() - t0:.0f} s", flush=True)
    res = {"n_recordings": len(recs), "k0_wbmae": s0, "k0_contrib": c0, "feature_check_max_abs": dev, "runs": {}}
    for name, k in KS.items():
        ks = [k if k is not None else int(rng.integers(0, max(int((onset[i] - 60) // 5), 0) + 1)) for i in range(len(recs))]
        ks = [min(kk, len(r) - 1) for kk, r in zip(ks, recs)]
        rows_list = [r[kk:] for r, kk in zip(recs, ks)]
        tr = [predict_from(r, kk) for r, kk in zip(recs, ks)]
        te = [predict_from(r, kk, true_elapsed=True) for r, kk in zip(recs, ks)]
        out = dict(k=ks if k is None else k, n_windows=int(sum(len(x) for x in rows_list)))
        for comp in ("mix", "hgb", "seq"):
            sr, cr = score(rows_list, [f[comp][kk:] for f, kk in zip(full, ks)])
            st, ct = score(rows_list, [f[comp] for f in tr])
            se, _ = score(rows_list, [f[comp] for f in te])
            out[comp] = dict(full_history=sr, truncated=st, truncated_true_elapsed=se, delta=st - sr, delta_true_elapsed=se - sr,
                             contrib_full=cr, contrib_truncated=ct, d_contrib=(np.array(ct) - np.array(cr)).tolist())
        res["runs"][name] = out
        m = out["mix"]
        print(f"{name:9s} windows {out['n_windows']:5d} | mix: full {m['full_history']:6.2f} truncated {m['truncated']:6.2f} "
              f"Δ {m['delta']:+6.2f}, with true elapsed Δ {m['delta_true_elapsed']:+6.2f} | hgb Δ {out['hgb']['delta']:+6.2f} "
              f"(true el {out['hgb']['delta_true_elapsed']:+6.2f}) | seq Δ {out['seq']['delta']:+6.2f} (true el "
              f"{out['seq']['delta_true_elapsed']:+6.2f}) | Δ contrib " + " ".join(f"{n} {x:+.2f}" for n, x in zip(BIN_NAMES, m["d_contrib"]))
              + (f" | k median {int(np.median(ks))}, max {max(ks)}" if k is None else ""), flush=True)
    res["time_s"] = time.perf_counter() - t0
    res["bundle"], res["gru_from"] = BUNDLE, GRU_FROM
    (OUT / f"stream_start_test{TAG}.json").write_text(json.dumps(res, indent=2))
    print(f"saved {OUT / f'stream_start_test{TAG}.json'} ({res['time_s']:.0f} s)")
