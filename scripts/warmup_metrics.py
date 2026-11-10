"""Warm-up leaderboard metrics (bMAE, MAE) next to W-bMAE on the public test set (40 recordings),
for models that never saw the test recordings.

Definitions (warm-up scoring program, 2026-competition/tracks/sleep_onset/objective.py):
- bmae: all test windows pooled; MAE inside true-target ranges [0,40) [40,90) [90,300) [300,600]
  ([lo,hi), 600 in the last), plain mean over non-empty ranges (equal weights, no per-recording step);
- mae : all test windows pooled, plain mean absolute error.
Same as neuralbench test/bmae (BinnedMAE, default edges, no weights) and test/mae.

Checks: bmae computed three ways (own numpy, competition binned_mae, neuralbench BinnedMAE torch);
EEGNet bmae/mae vs neuralbench's logged test/bmae, test/mae; final540 vs its Docker scoring
output (final540 is used ONLY for that check - it saw the test recordings).

Usage: python scripts/warmup_metrics.py   (writes ~/neuralbench/save/warmup_metrics.json)
"""
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ablation_c import ABLATIONS, build_table_ext
from compare_final import parts
from cv_utils import apply_bin_edges
from dist_hgb import decide
from stream_utils import EDGES, load_window_table, summarize, weighted_median, window_weights
from train_hgb import SAVE

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compet_metrics", Path.home() / "projects" / "2026-competition" / "benchmark_utils" / "metrics.py")
compet_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compet_metrics)
from neuralbench.metrics import BinnedMAE  # noqa: E402


def bmae_own(y, p):
    rng = np.clip(np.searchsorted(EDGES, y, side="right") - 1, 0, 3)
    err = np.abs(p - y)
    return float(np.mean([err[rng == b].mean() for b in range(4) if (rng == b).any()]))


def metrics(d, p):
    y, p = d.target.to_numpy(np.float64), np.asarray(p, np.float64)
    m = BinnedMAE()
    m.update(torch.tensor(p), torch.tensor(y))
    b = dict(own=bmae_own(y, p), compet=compet_metrics.binned_mae(y, p),
             neuralbench=float(m.compute()))
    assert max(b.values()) - min(b.values()) < 1e-4, b  # neuralbench returns float32
    return dict(wbmae=summarize(d, p)["wbmae"], bmae=b["compet"], mae=float(np.abs(p - y).mean()),
                bmae_ways=b)


df = load_window_table()
train = df[df.split == "train"].reset_index(drop=True)
out, checks = {}, {}

# 1. constant (train weighted median, as in simple_baselines.py)
wb, sb, _ = ABLATIONS["dyn_maxmin_spec_eye"]
dfx, Xx, _ = build_table_ext(wb, sb)  # same row order as features; split/target/timeline included
mt = (dfx.split == "test").to_numpy()
d_te = dfx[mt].reset_index(drop=True)
c = weighted_median(train.target.to_numpy(), window_weights(train))
out["constant"] = metrics(d_te, np.full(len(d_te), c)) | dict(note=f"{c:.1f} s")

# 2. elapsed rule (the exported/Docker-tested table, fitted on train)
tab = json.loads((ROOT / "submissions" / "elapsed_rule" / "elapsed_table.json").read_text())
vals, bin_s = np.array(tab["values_s"]), tab["bin_s"]
idx = np.minimum(np.floor((d_te.elapsed_s.to_numpy() - 1e-9) / bin_s).astype(int), len(vals) - 1)
out["elapsed_rule"] = metrics(d_te, vals[idx])

# 3. EEGNet, 3 seeds: re-run checkpoints on the NeuralBench test loader
from neuralbench.experiment_config import build_experiment_configs  # noqa: E402
from neuralbench.main import Experiment  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
configs = {c_["seed"]: c_ for c_ in build_experiment_configs("eeg", "_sleep_onset_stream", model="eegnet", quiet=True)}
loaders, eeg = None, []
for f in sorted((SAVE / "eegnet_baseline").glob("seed*.json")):
    s = json.loads(f.read_text())
    exp = Experiment(**configs[s["seed"]])
    if loaders is None:
        loaders = exp.data.prepare()
    exp.prepare_pl_module(loaders["train"], loaders["val"])
    mod = exp._brain_module
    mod.load_state_dict(torch.load(s["checkpoint"], map_location="cpu", weights_only=True)["state_dict"])
    mod.to(dev).eval()
    ds = loaders["test"].dataset
    preds, tgts, tls = [], [], []
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=256, shuffle=False, collate_fn=ds.collate_fn):
            batch = batch.to(dev)
            preds.append(mod.model_forward(batch).reshape(-1).float().cpu().numpy())
            tgts.append(batch.data["target"].reshape(-1).float().cpu().numpy())
            tls += [sg.timeline for sg in batch.segments]
    d = pd.DataFrame({"timeline": tls, "target": np.concatenate(tgts).astype(np.float64)})
    r = metrics(d, np.concatenate(preds))
    logged = s["all_test_metrics"]
    checks[f"eegnet_seed{s['seed']}"] = dict(
        bmae_ours=r["bmae"], bmae_neuralbench_logged=logged["test/bmae"],
        mae_ours=r["mae"], mae_neuralbench_logged=logged["test/mae"],
        wbmae_ours=r["wbmae"], wbmae_neuralbench_logged=logged["test/wbmae_stream_mean"])
    eeg.append(r)
out["eegnet_3seeds"] = {k: float(np.mean([r[k] for r in eeg])) for k in ("wbmae", "bmae", "mae")} | dict(
    per_seed={k: [r[k] for r in eeg] for k in ("wbmae", "bmae", "mae")})

# 4. hgb_v3 (B + dyn, embedded calibration = hgb_v3_np.zip)
v3 = joblib.load(SAVE / "hgb_v3" / "model.joblib")
dfv, Xv, _ = build_table_ext((), v3["groups"])
mv = (dfv.split == "test").to_numpy()
P = v3["model"].predict_proba(apply_bin_edges(Xv[mv], v3["bin_edges"]))
raw = decide(P, v3["reps"], v3["omega"], True, v3["class_edges"])
out["hgb_v3"] = metrics(dfv[mv].reset_index(drop=True), np.clip(np.interp(raw, v3["calib_x"], v3["calib_y"]), 0, 600))

# 5. mix_gru10_hgb (500 recordings) + final540 (Docker check only)
zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
F = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)
for name, key in (("mix_uw_n10", "mix_gru10_hgb"), ("mix_uw_n10_all", "final540")):
    _, _, p, d = parts(joblib.load(SAVE / name / "model.joblib"), dfx, Xx, F, mt)
    r = metrics(d, p)
    if key == "final540":
        checks["final540_vs_docker"] = dict(bmae_ours=r["bmae"], bmae_docker=83.14301434090218,
                                            mae_ours=r["mae"], mae_docker=157.9322509765625,
                                            wbmae_ours=r["wbmae"], wbmae_docker=31.533730839718874)
    else:
        out[key] = r

print(f"{'model':16s} {'W-bMAE':>8s} {'bMAE':>8s} {'MAE':>8s}")
for k, v in out.items():
    print(f"{k:16s} {v['wbmae']:8.3f} {v['bmae']:8.3f} {v['mae']:8.3f}")
print("checks:")
for k, v in checks.items():
    print(" ", k, {a: round(b, 4) for a, b in v.items()})
(SAVE / "warmup_metrics.json").write_text(json.dumps(dict(models=out, checks=checks), indent=2))
