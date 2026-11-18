"""Final three-way sequence mix: (GRU-uw + TCN + GRU-emb) / 3 on the metric-weighted grid + HGB(c),
alpha = 0.7 (dual CV 39.40, nested-LOFO alpha 0.7 in all 10 folds; scripts/mix_seq.py).

Sequence models trained on all train+val (or all 540 with --all) with fixed epochs from the dual CV
(cosine schedule unchanged, stopped at that epoch):
  GRU-uw   10 seeds x 15 epochs (as final540)
  TCN      10 seeds x 10 epochs (CV mean curve: epoch 10 41.7, epoch 15 42.2)
  CNN enc   3 seeds x 12 epochs (LOFO epochs 9-15, median 11.5); embeddings for all training windows
  GRU-emb   3 seeds x 10 epochs (GRU-uw + 32-d embedding of the same seed's encoder; LOFO 10 in 8/10)
HGB part = ens_c (train+val) or ens_c_all<tag> (540). Test is scored once as a pipeline check
(in-sample with --all).

Usage: python scripts/train_mix3_final.py [--all] [--ens-tag=X] [--tag=T] [--deterministic] [--crop] [--epochs=gru:20,tcn:15,gemb:10]
  --crop: stream-start crop augmentation of the sequence models (gru_cv.run_fold cfg["crop"] = 50/35/15 %); --epochs overrides
  the fixed epochs of the plan (from the dual CV LOFO choices of the augmented runs).
  writes ~/neuralbench/save/mix3<T>[_all<X>]/model.joblib + results.json
  --deterministic: cuDNN deterministic algorithms (repro.set_deterministic); without it the CNN encoder is not
  bit-reproducible on GPU (scripts/check_determinism.py). mix3 / mix3_all were trained without it.
"""
import json
import sys
import time
from pathlib import Path

if "--deterministic" in sys.argv:
    from repro import set_deterministic
    set_deterministic()

import joblib
import numpy as np
import torch

from ablation_c import ABLATIONS, build_table_ext
from cnn_encoder import CLIP, EMB, RAW, SCALE, Encoder, infer
from cnn_encoder import train_fold as train_encoder
from cv_utils import apply_bin_edges
from dist_hgb import VARIANTS, class_stats, decide, to_class
from ensemble_dist import GRID, to_grid
from gru_cv import ARCHS, CONFIGS, DEV, load_inputs, pad, run_fold, seq_inputs
from stream_utils import RANGE_NAMES, summarize
from train_hgb import SAVE

ALL = "--all" in sys.argv
ENS_TAG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ens-tag=")), "")
TAG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "")
ENS_DIR = SAVE / (f"ens_c_all{ENS_TAG}" if ALL else "ens_c")
OUT = SAVE / (f"mix3{TAG}_all{ENS_TAG}" if ALL else f"mix3{TAG}")
OUT.mkdir(parents=True, exist_ok=True)
B = VARIANTS["B"]
ALPHA = 0.7
PLAN = dict(gru=dict(cfg="uw", seeds=range(10), epochs=15), tcn=dict(cfg="tcn_n10", seeds=range(10), epochs=10),
            cnn=dict(seeds=range(3), epochs=12), gemb=dict(cfg="uw", seeds=range(3), epochs=10))
CROP = (0.5, 0.35, 0.15) if "--crop" in sys.argv else None
if "--cropE" in sys.argv:  # Work E (18 Nov): crop mix gru_cv.CROP_E
    from gru_cv import CROP_E
    CROP = CROP_E
for a in sys.argv:
    if a.startswith("--epochs="):
        for kv in a.split("=", 1)[1].split(","):
            k, v = kv.split(":")
            PLAN[k]["epochs"] = int(v)
def seq_cfg(name):
    c = CONFIGS[name]
    return dict(c, crop=CROP) if CROP else c


def states_of(model):
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


# ---- data
d, recs, F, gain = load_inputs(include_test=ALL)
tr = np.ones(len(d), bool)
_, omega_b = class_stats(d, B)
split = np.load(SAVE / "features_base.npz")["split"]
mask = np.ones(len(split), bool) if ALL else (split != "test")
raw = np.load(RAW, mmap_mode="r")
X_raw = torch.as_tensor(np.asarray(raw[mask]), device=DEV)
y_cls = torch.as_tensor(to_class(d.target.to_numpy(), B), device=DEV).long()
print(f"{'all 540' if ALL else 'train+val'}: {len(d)} windows, {len(recs)} recordings; HGB from {ENS_DIR}", flush=True)
timing = {}

# ---- GRU-uw and TCN (same 125 inputs)
seq_states, seq_norm = {}, {}
for name in ("gru", "tcn"):
    p, cfg = PLAN[name], seq_cfg(PLAN[name]["cfg"])
    t0 = time.perf_counter()
    seq_states[name] = []
    for s in p["seeds"]:
        model, mu, sd = run_fold(d, recs, F, gain, tr, None, cfg, s, stop_epoch=p["epochs"])
        seq_states[name].append(states_of(model))
    seq_norm[name] = (mu, sd)
    timing[name] = time.perf_counter() - t0
    print(f"{name}: {len(p['seeds'])} seeds x {p['epochs']} epochs in {timing[name]:.0f} s", flush=True)

# ---- CNN encoder (3 seeds) -> embeddings of all training windows -> GRU-emb
t0 = time.perf_counter()
cnn_states, embs = [], []
for s in PLAN["cnn"]["seeds"]:
    enc = train_encoder(X_raw, y_cls, tr, None, s, stop_epoch=PLAN["cnn"]["epochs"])
    cnn_states.append(states_of(enc))
    _, e = infer(enc, X_raw, torch.arange(len(d), device=DEV))
    embs.append(e.cpu().numpy().astype(np.float32))
timing["cnn"] = time.perf_counter() - t0
print(f"cnn: {len(cnn_states)} seeds x {PLAN['cnn']['epochs']} epochs in {timing['cnn']:.0f} s", flush=True)
t0 = time.perf_counter()
gemb_states = []
cfg_g = seq_cfg(PLAN["gemb"]["cfg"])
for s, e in zip(PLAN["gemb"]["seeds"], embs):
    model, mu_e, sd_e = run_fold(d, recs, F, gain, tr, None, cfg_g, s, stop_epoch=PLAN["gemb"]["epochs"], extra=e)
    gemb_states.append(states_of(model))
timing["gemb"] = time.perf_counter() - t0
print(f"gemb: {len(gemb_states)} seeds x {PLAN['gemb']['epochs']} epochs in {timing['gemb']:.0f} s", flush=True)

# ---- test check (once): batched inference, all components
ens = joblib.load(ENS_DIR / "model.joblib")
wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, X, _ = build_table_ext(wg, sg)
mt = (df.split == "test").to_numpy()
d_te = df[mt].reset_index(drop=True)
zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
Fte = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)[mt]
te_recs = [g.sort_values("stop").index.to_numpy() for _, g in d_te.groupby("timeline", sort=False)]
X_te_raw = torch.as_tensor(np.asarray(raw[mt]), device=DEV)


def seq_probs(arch, states, mu, sd, cfg, extra=None):
    out = []
    for sdict in states:
        net = ARCHS[arch](len(mu), len(B) - 1, cfg["hidden"], cfg["dropout"], cfg["in_dropout"]).to(DEV)
        net.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
        net.eval()
        xt = torch.tensor(pad([(seq_inputs(Fte[r], None if extra is None else extra[r]) - mu) / sd for r in te_recs]),
                          device=DEV)
        with torch.no_grad():
            P = torch.softmax(net(xt)[0], -1).cpu().numpy()
        probs = np.empty((len(d_te), len(B) - 1), np.float32)
        for i, r in enumerate(te_recs):
            probs[r] = P[i, :len(r)]
        out.append(probs)
    return np.mean(out, axis=0).astype(np.float64)


def weighted_grid(P):
    q = P * omega_b
    return to_grid(q / q.sum(1, keepdims=True), B)


Pg = seq_probs("gru", seq_states["gru"], *seq_norm["gru"], CONFIGS["uw"])
Pt = seq_probs("tcn", seq_states["tcn"], *seq_norm["tcn"], CONFIGS["tcn_n10"])
Pe = []
for sdict, st in zip(cnn_states, gemb_states):
    enc = Encoder().to(DEV)
    enc.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
    enc.eval()
    _, e_te = infer(enc, X_te_raw, torch.arange(len(d_te), device=DEV))
    Pe.append(seq_probs("gru", [st], mu_e, sd_e, cfg_g, extra=e_te.cpu().numpy().astype(np.float32)))
Pe = np.mean(Pe, axis=0)
qg, qt, qe = weighted_grid(Pg), weighted_grid(Pt), weighted_grid(Pe)
Xb = apply_bin_edges(X[mt], ens["bin_edges"])
Wh = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(ens["models"], ens["member_edges"])], axis=0)
Wh = Wh * ens["grid_omega"]
Wh /= Wh.sum(1, keepdims=True)
ones = np.ones(len(GRID) - 1)
q3 = (qg + qt + qe) / 3
q2 = (qg + qt) / 2
cands = {"gru": qg, "tcn": qt, "gemb": qe, "hgb_c_plain": Wh, "mix2_a0.7": 0.7 * q2 + 0.3 * Wh,
         "mix3_a0.7": ALPHA * q3 + (1 - ALPHA) * Wh}
preds = {k: decide(W, None, ones, True, GRID) for k, W in cands.items()}
scores = {k: summarize(d_te, p) for k, p in preds.items()}
for k, v in scores.items():
    print(f"test{' (in-sample)' if ALL else ''} {k:12s} {v['wbmae']:.4f}  " + " ".join(f"{r} {v[r]:.1f}" for r in RANGE_NAMES))
np.savez(OUT / "test_preds.npz", timeline=d_te.timeline.to_numpy(), stop=d_te.stop.to_numpy(), **preds)

bundle = dict(ens)
bundle.update(kind="mix3", calib_x=None, calib_y=None,
              gru_states=seq_states["gru"], gru_mu=seq_norm["gru"][0], gru_sd=seq_norm["gru"][1],
              gru_cfg=dict(CONFIGS["uw"]), gru_weighted=False, gru_omega_b=omega_b, gru_edges=B,
              tcn_states=seq_states["tcn"], tcn_mu=seq_norm["tcn"][0], tcn_sd=seq_norm["tcn"][1],
              tcn_dilations=(1, 2, 4, 8, 16, 32),
              cnn_states=cnn_states, cnn_scale=SCALE, cnn_clip=CLIP, cnn_emb=EMB,
              gemb_states=gemb_states, gemb_mu=mu_e, gemb_sd=sd_e,
              mix_alpha=ALPHA, gru_baseline_n=24, gru_elapsed_clip=1500.0)
joblib.dump(bundle, OUT / "model.joblib")
(OUT / "results.json").write_text(json.dumps(dict(
    all540=ALL, deterministic="--deterministic" in sys.argv, crop=CROP, n_train_windows=len(d), ens_dir=str(ENS_DIR), alpha=ALPHA,
    plan={k: dict(v, seeds=list(v["seeds"])) for k, v in PLAN.items()}, timing_s=timing, test=scores), indent=2))
print("saved", OUT, flush=True)
