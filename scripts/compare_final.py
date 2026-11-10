"""Checks for the final 540-recording model (docs/final_plan.md, step 3).

1. 500-model vs 540-model: mixed predictions on the same windows (train+val and test separately):
   mean |diff|, 95th pct |diff|, correlation, W-bMAE of each (500 on test = out-of-sample; everything
   else in-sample -> sanity only). Also per-part (GRU / HGB) diffs.
2. Bit-level reproducibility: two 540 runs with the same seeds -> max |diff| of HGB probabilities,
   GRU weights, mixed predictions.

Usage: python scripts/compare_final.py MIX500_DIR MIX540_DIR [MIX540_REPEAT_DIR]
  (dirs relative to ~/neuralbench/save, e.g. mix_uw_n10 mix_uw_n10_all mix_uw_n10_all_rep)
"""
import json
import sys

import joblib
import numpy as np
import torch

from ablation_c import ABLATIONS, build_table_ext
from cv_utils import apply_bin_edges
from dist_hgb import decide
from ensemble_dist import GRID, to_grid
from gru_cv import DEV, GRUNet, pad, seq_inputs
from stream_utils import summarize
from train_hgb import SAVE


def parts(bundle, df, X, F, mask):
    """-> (GRU grid distribution, HGB grid distribution, mixed prediction in s) for rows df[mask]."""
    d = df[mask].reset_index(drop=True)
    Fm = F[mask]
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    cfg, B = bundle["gru_cfg"], bundle["gru_edges"]
    xin = [(seq_inputs(Fm[r]) - bundle["gru_mu"]) / bundle["gru_sd"] for r in recs]
    Pq = []
    for sdict in bundle["gru_states"]:
        net = GRUNet(len(bundle["gru_mu"]), len(B) - 1, cfg["hidden"], cfg["dropout"], cfg["in_dropout"]).to(DEV)
        net.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
        net.eval()
        probs = np.empty((len(d), len(B) - 1), np.float32)
        for i0 in range(0, len(recs), 64):  # chunks of recordings (memory)
            xt = torch.tensor(pad(xin[i0:i0 + 64]), device=DEV)
            with torch.no_grad():
                P = torch.softmax(net(xt)[0], -1).cpu().numpy()
            for i, r in enumerate(recs[i0:i0 + 64]):
                probs[r] = P[i, :len(r)]
        Pq.append(probs)
    q = np.mean(Pq, axis=0).astype(np.float64)
    if not bundle["gru_weighted"]:
        q = q * bundle["gru_omega_b"]
    q = to_grid(q / q.sum(1, keepdims=True), B)
    Xb = apply_bin_edges(X[mask], bundle["bin_edges"])
    Wh = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(bundle["models"], bundle["member_edges"])], axis=0)
    Wh = Wh * bundle["grid_omega"]
    Wh /= Wh.sum(1, keepdims=True)
    a = bundle["mix_alpha"]
    ones = np.ones(len(GRID) - 1)
    pred = decide(a * q + (1 - a) * Wh, None, ones, True, GRID)
    return q, Wh, pred, d


def diff_stats(a, b):
    ad = np.abs(a - b)
    return dict(mean_abs=float(ad.mean()), p95_abs=float(np.percentile(ad, 95)), max_abs=float(ad.max()),
                corr=float(np.corrcoef(a, b)[0, 1]))


def main():
    dirs = sys.argv[1:]
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X, _ = build_table_ext(wg, sg)
    zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
    F = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)
    masks = {"trainval": (df.split != "test").to_numpy(), "test": (df.split == "test").to_numpy()}
    bundles = {k: joblib.load(SAVE / k / "model.joblib") for k in dirs}
    out = {}
    res = {k: {s: parts(b, df, X, F, m) for s, m in masks.items()} for k, b in bundles.items()}

    m500, m540 = dirs[0], dirs[1]
    print(f"== 500 ({m500}) vs 540 ({m540}) ==")
    for s in masks:
        q5, h5, p5, d = res[m500][s]
        q4, h4, p4, _ = res[m540][s]
        st = diff_stats(p5, p4)
        st["gru_dist_mean_abs"] = float(np.abs(q5 - q4).sum(1).mean() / 2)  # mean total-variation distance
        st["hgb_dist_mean_abs"] = float(np.abs(h5 - h4).sum(1).mean() / 2)
        st["wbmae_500"] = summarize(d, p5)["wbmae"]
        st["wbmae_540"] = summarize(d, p4)["wbmae"]
        out[f"500_vs_540_{s}"] = st
        print(f"{s:9s} n={len(d):6d}  mean|d| {st['mean_abs']:.2f} s  p95 {st['p95_abs']:.2f}  max "
              f"{st['max_abs']:.1f}  corr {st['corr']:.4f}  TV(GRU) {st['gru_dist_mean_abs']:.4f}  "
              f"TV(HGB) {st['hgb_dist_mean_abs']:.4f}  W-bMAE 500 {st['wbmae_500']:.4f}  540 {st['wbmae_540']:.4f}")

    if len(dirs) > 2:
        r = dirs[2]
        a, b = bundles[m540], bundles[r]
        gw = max(float(np.abs(x[k] - y[k]).max()) for x, y in zip(a["gru_states"], b["gru_states"]) for k in x)
        bins_eq = all(np.array_equal(x, y) for x, y in zip(a["bin_edges"], b["bin_edges"]))
        rep = dict(gru_weights_max_abs=gw, bin_edges_equal=bool(bins_eq),
                   mu_sd_max_abs=float(max(np.abs(a["gru_mu"] - b["gru_mu"]).max(), np.abs(a["gru_sd"] - b["gru_sd"]).max())))
        for s in masks:
            q1, h1, p1, _ = res[m540][s]
            q2, h2, p2, _ = res[r][s]
            rep[f"{s}_gru_dist_max_abs"] = float(np.abs(q1 - q2).max())
            rep[f"{s}_hgb_dist_max_abs"] = float(np.abs(h1 - h2).max())
            rep[f"{s}_pred_max_abs"] = float(np.abs(p1 - p2).max())
        out["repeat"] = rep
        print(f"== repeat ({m540} vs {r}) ==")
        for k, v in rep.items():
            print(f"  {k}: {v}")

    (SAVE / m540 / "compare.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
