"""Final GRU + HGB(c) mix: GRU (config CFG, SEEDS) trained on all of train+val for a fixed number of
epochs taken from the dual CV (same cosine schedule, stopped at that epoch), HGB part = the already
trained ensemble (c) (~/neuralbench/save/ens_c/model.joblib). Mix weight alpha from the dual CV.
No calibration (it hurt the mix in CV). Test is scored once, as a final check only.

Usage: python scripts/train_mix_final.py CFG N_EPOCHS ALPHA [SEEDS] [TAG]
  SEEDS: comma list (default gru_cv.SEEDS = 0,1,2); writes ~/neuralbench/save/mix_<CFG><TAG>/
"""
import json
import sys
import time

import joblib
import numpy as np
import torch

from ablation_c import ABLATIONS, build_table_ext
from cv_utils import apply_bin_edges
from dist_hgb import VARIANTS, class_stats, decide
from ensemble_dist import GRID, to_grid
from gru_cv import CONFIGS, DEV, SEEDS, load_inputs, pad, run_fold, seq_inputs
from stream_utils import RANGE_NAMES, summarize
from train_hgb import SAVE

cfg_name, n_epochs, alpha = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
if len(sys.argv) > 4:
    SEEDS = tuple(int(x) for x in sys.argv[4].split(","))
TAG = sys.argv[5] if len(sys.argv) > 5 else ""
cfg = CONFIGS[cfg_name]
OUT = SAVE / f"mix_{cfg_name}{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
B = VARIANTS["B"]

# ---- GRU on all train+val
d, recs, F, gain = load_inputs()
tr = np.ones(len(d), bool)
_, omega_b = class_stats(d, B)
t0 = time.perf_counter()
grus = []
for s in SEEDS:
    model, mu, sd = run_fold(d, recs, F, gain, tr, None, cfg, s, stop_epoch=n_epochs)
    grus.append({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
fit_s = time.perf_counter() - t0
print(f"GRU {cfg_name}: {len(SEEDS)} seeds x {n_epochs} epochs in {fit_s:.0f} s", flush=True)

# ---- test check (once): GRU probs + HGB(c) grid probs, mixed
ens = joblib.load(SAVE / "ens_c" / "model.joblib")
wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, X, _ = build_table_ext(wg, sg)
mt = (df.split == "test").to_numpy()
d_te = df[mt].reset_index(drop=True)
zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
Fall = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)[mt]
te_recs = [g.sort_values("stop").index.to_numpy() for _, g in d_te.groupby("timeline", sort=False)]
from gru_cv import GRUNet
Pq = []
for sdict in grus:
    net = GRUNet(len(mu), len(B) - 1, cfg["hidden"], cfg["dropout"], cfg["in_dropout"]).to(DEV)
    net.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
    net.eval()
    xt = torch.tensor(pad([(seq_inputs(Fall[r]) - mu) / sd for r in te_recs]), device=DEV)
    with torch.no_grad():
        P = torch.softmax(net(xt)[0], -1).cpu().numpy()
    probs = np.empty((len(d_te), len(B) - 1), np.float32)
    for i, r in enumerate(te_recs):
        probs[r] = P[i, :len(r)]
    Pq.append(probs)
q = np.mean(Pq, axis=0).astype(np.float64)
if not cfg["weighted"]:
    q = q * omega_b
q = to_grid(q / q.sum(1, keepdims=True), B)
Xb = apply_bin_edges(X[mt], ens["bin_edges"])
Wh = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(ens["models"], ens["member_edges"])], axis=0)
Wh = Wh * ens["grid_omega"]
Wh /= Wh.sum(1, keepdims=True)
ones = np.ones(len(GRID) - 1)
scores = {k: summarize(d_te, decide(W, None, ones, True, GRID)) for k, W in
          (("gru", q), ("hgb_c_plain", Wh), ("mix", alpha * q + (1 - alpha) * Wh))}
for k, v in scores.items():
    print(f"test {k:12s} {v['wbmae']:.4f}  " + " ".join(f"{r} {v[r]:.1f}" for r in RANGE_NAMES))

bundle = dict(ens)
bundle.update(kind="mix", calib_x=None, calib_y=None, gru_states=grus, gru_mu=mu, gru_sd=sd,
              gru_cfg=dict(cfg), gru_weighted=cfg["weighted"], gru_omega_b=omega_b, gru_edges=B,
              mix_alpha=alpha, gru_baseline_n=24, gru_elapsed_clip=1500.0)
joblib.dump(bundle, OUT / "model.joblib")
(OUT / "results.json").write_text(json.dumps(dict(cfg=cfg_name, n_epochs=n_epochs, alpha=alpha, seeds=list(SEEDS),
                                                  fit_s=fit_s, test=scores), indent=2))
