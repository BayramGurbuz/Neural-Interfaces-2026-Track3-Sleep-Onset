"""Comparison only: re-decide mix_gru10_hgb (500 recordings) test distributions with the decision
that is optimal for the warm-up bMAE instead of W-bMAE. The model (GRU + HGB probabilities, alpha)
is unchanged; only the class weights omega in "weighted median of P * omega" change:

- W-bMAE (submitted): omega_k = mean over train+val windows of class k of the W-bMAE window weight
  (10/5/3/1 per range, normalized inside each recording, recordings averaged).
- bMAE: pooled window weight w_i = 1 / (4 * N_range(i)) (4 ranges equal, normalized by the number of
  pooled windows in the range, no per-recording step) -> omega_k = 1 / (4 * N_range(k)).
  N_range from train+val (a decision rule may only use training data); an "oracle" variant with
  N_range from the test pool itself is shown for sensitivity only.

Usage: python scripts/bmae_decision.py
"""
import importlib.util
from pathlib import Path

import joblib
import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from compare_final import parts
from dist_hgb import to_class
from ensemble_dist import GRID
from stream_utils import RANGE_NAMES, range_index, summarize
from train_hgb import SAVE

spec = importlib.util.spec_from_file_location(
    "compet_metrics", Path.home() / "projects" / "2026-competition" / "benchmark_utils" / "metrics.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)


def bmae_omega(target, edges):
    """Class weights for the pooled, equal-range bMAE (normalized to sum 1)."""
    rng = range_index(target)
    n_r = np.bincount(rng, minlength=4).astype(float)
    w = 1.0 / (4.0 * n_r[rng])
    c = to_class(target, edges)
    om = np.array([w[c == k].mean() if (c == k).any() else 0.0 for k in range(len(edges) - 1)])
    return om / om.sum()


wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, X, _ = build_table_ext(wg, sg)
zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
F = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)
mt = (df.split == "test").to_numpy()
bundle = joblib.load(SAVE / "mix_uw_n10" / "model.joblib")
B = bundle["gru_edges"]

variants = {"W-bMAE kararı (gönderilen)": bundle}
for name, pool in (("bMAE kararı (N_aralık train+val)", ~mt), ("bMAE kararı, oracle (N_aralık test)", mt)):
    t = df.target.to_numpy()[pool]
    variants[name] = dict(bundle, gru_omega_b=bmae_omega(t, B), grid_omega=bmae_omega(t, GRID))

print(f"{'karar':38s} {'W-bMAE':>8s} {'bMAE':>8s} {'MAE':>8s}   havuz aralık MAE " + " ".join(RANGE_NAMES))
for name, b in variants.items():
    _, _, p, d = parts(b, df, X, F, mt)
    y = d.target.to_numpy(np.float64)
    rng = range_index(y)
    per = [np.abs(p - y)[rng == r].mean() for r in range(4)]
    print(f"{name:38s} {summarize(d, p)['wbmae']:8.3f} {cm.binned_mae(y, p):8.3f} {np.abs(p - y).mean():8.3f}   "
          + " ".join(f"{v:7.1f}" for v in per)
          + f"   ort. tahmin {p.mean():.1f} s")
print("havuz pencere sayıları (test):", np.bincount(range_index(df.target.to_numpy()[mt]), minlength=4),
      " train+val:", np.bincount(range_index(df.target.to_numpy()[~mt]), minlength=4))
