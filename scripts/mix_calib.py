"""Experiments 1-2 on the GRU(10 seeds) + HGB(c) mix: distribution temperature and region-wise alpha.

Candidates: M = alpha * q_GRU + (1 - alpha) * W_HGB (both metric-weighted grid distributions, as in
mix_gru_hgb.py), tempered M_tau ∝ M ** (1 / tau) (tau < 1 sharpens, > 1 flattens), then the same
decision (continuous weighted median, omega = 1). Grid: alpha 0..1 step 0.1, tau in TAUS.

Selection is nested LOFO over all 10 dual-CV folds (5 unseen + 5 seen): for held-out fold f the
parameters minimise the summed W-bMAE loss of the other 9 folds, with every recording that belongs
to fold f masked out of those folds (seen and unseen folds share recordings). W-bMAE is a weighted
sum of window errors once the true targets are fixed (stream_utils.window_weights), so a per-region
choice is separable: each region gets its own argmin.

Regions (all causal at inference):
- "pred": metric range [0,40) [40,90) [90,300) [300,600] of the fixed alpha 0.6, tau 1 mix prediction
  (the final540 recipe), so the region does not depend on the parameters being chosen;
- "elapsed": 3 bins of elapsed time, boundaries = tertiles of the fold's training windows.

Methods: ref (per-protocol LOFO alpha = mix_uw_n10), base (alpha 0.6, tau 1), A_global, T_global,
T_pred, A_pred, A_elapsed, AT_pred (joint per region), A_pred+T_global, A_elapsed+T_global.

Usage: python scripts/mix_calib.py   (writes OUT/mixcal_<method>_{unseen,seen}_preds.npz, mix_calib.json)
"""
import json

import numpy as np

from dist_hgb import VARIANTS, class_stats, decide
from dual_cv import OUT, load_trainval, seen_splits, unseen_splits
from ensemble_dist import GRID, MEMBERS, grid_omega, to_grid
from range_breakdown import breakdown, load_preds, print_breakdown
from stream_utils import range_index, summarize, window_weights

FEAT_SET = "dyn_maxmin_spec_eye"
HGB_MEMBERS = ["B", "B_h1", "B_h2", "B2"]
GRU = "uw_n10"
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
TAUS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
BASE_ALPHA, BASE_TAU = 0.6, 1.0
ONES = np.ones(len(GRID) - 1)


def components(d, protocols):
    """Per fold (protocol order unseen, seen): dict with q, Wh (grid, metric-weighted), dte, tr mask."""
    hgb = {m: np.load(OUT / f"ens_{FEAT_SET}_{m}.npz", allow_pickle=True)["probas"].item() for m in HGB_MEMBERS}
    folds = []
    for proto, splits in protocols.items():
        z = np.load(OUT / f"gru_{GRU}_{proto}_proba.npz")
        for k, (tr, te) in enumerate(splits):
            dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
            Wh = np.mean([to_grid(hgb[m][proto][k], MEMBERS[m][0]) for m in HGB_MEMBERS], axis=0) * grid_omega(dtr)
            Wh /= Wh.sum(1, keepdims=True)
            _, ob = class_stats(dtr, VARIANTS["B"])
            q = z[f"arr_{k}"].astype(np.float64) * ob
            q = to_grid(q / q.sum(1, keepdims=True), VARIANTS["B"])
            folds.append(dict(proto=proto, k=k, q=q, Wh=Wh, dte=dte, dtr=dtr))
    return folds


def predict(f, alpha, tau):
    M = alpha * f["q"] + (1 - alpha) * f["Wh"]
    if tau != 1.0:
        M = np.power(np.maximum(M, 0), 1.0 / tau)
    return decide(M, None, ONES, True, GRID)


def nested_select(folds, cand_preds, regions, n_regions):
    """cand_preds[f]: (n_cand, n_win); regions[f]: (n_win,) ints. Returns preds per fold, choices."""
    n = len(folds)
    recs = [f["dte"].timeline.to_numpy() for f in folds]
    ys = [f["dte"].target.to_numpy() for f in folds]
    ws = [window_weights(f["dte"]) for f in folds]
    E = [np.abs(cand_preds[g] - ys[g][None]) * ws[g][None] for g in range(n)]  # (n_cand, n_win)
    out, choices = [], []
    for f in range(n):
        held = set(recs[f])
        L = np.zeros((cand_preds[f].shape[0], n_regions))
        for g in range(n):
            if g == f:
                continue
            keep = ~np.isin(recs[g], list(held))
            for r in range(n_regions):
                m = keep & (regions[g] == r)
                L[:, r] += E[g][:, m].sum(1)
        c = L.argmin(0)  # per region
        choices.append(c.tolist())
        out.append(cand_preds[f][c[regions[f]], np.arange(len(ys[f]))].astype(np.float32))
    return out, choices


def to_protocols(folds, preds):
    res = {"unseen": [], "seen": []}
    for f, p in zip(folds, preds):
        res[f["proto"]].append(p)
    return res


if __name__ == "__main__":
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    folds = components(d, protocols)

    # reproduce the reference (per-protocol LOFO alpha, tau 1) and check against the saved OOF
    ref_saved = load_preds("mix_uw_n10", protocols)
    ref = {}
    for proto, splits in protocols.items():
        fs = [f for f in folds if f["proto"] == proto]
        C = np.array([[summarize(f["dte"], predict(f, a, 1.0))["wbmae"] for a in ALPHAS] for f in fs])
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(fs))]
        ref[proto] = [predict(f, ALPHAS[j], 1.0).astype(np.float32) for f, j in zip(fs, sel)]
        dmax = max(float(np.abs(a - b).max()) for a, b in zip(ref[proto], ref_saved[proto]))
        print(f"reference {proto}: alphas {[float(ALPHAS[j]) for j in sel]}, max |Δ| vs saved = {dmax:.2e}", flush=True)
    ref_b = breakdown(d, protocols, ref_saved)

    # candidate predictions: full alpha x tau grid per fold
    pairs = [(a, t) for a in ALPHAS for t in TAUS]
    idx = {p: i for i, p in enumerate(pairs)}
    cand = [np.stack([predict(f, a, t) for a, t in pairs]) for f in folds]
    base_i = idx[(BASE_ALPHA, BASE_TAU)]
    reg_pred = [range_index(c[base_i]) for c in cand]
    reg_el = []
    for f in folds:
        b = np.quantile(f["dtr"].elapsed_s.to_numpy(), [1 / 3, 2 / 3])
        reg_el.append(np.searchsorted(b, f["dte"].elapsed_s.to_numpy(), side="right"))
    zero = [np.zeros(len(f["dte"]), int) for f in folds]

    def sub(cols):
        return [c[cols] for c in cand]

    a_idx = [idx[(a, 1.0)] for a in ALPHAS]
    t_idx = [idx[(BASE_ALPHA, t)] for t in TAUS]
    methods = {}
    methods["base"] = ([c[base_i].astype(np.float32) for c in cand], None)
    methods["A_global"] = nested_select(folds, sub(a_idx), zero, 1)
    methods["T_global"] = nested_select(folds, sub(t_idx), zero, 1)
    methods["T_pred"] = nested_select(folds, sub(t_idx), reg_pred, 4)
    methods["A_pred"] = nested_select(folds, sub(a_idx), reg_pred, 4)
    methods["A_elapsed"] = nested_select(folds, sub(a_idx), reg_el, 3)
    methods["AT_pred"] = nested_select(folds, cand, reg_pred, 4)

    # region alpha + one global tau: for each tau the per-region alpha is separable; pick tau by total loss
    def region_alpha_global_tau(regions, n_regions):
        recs = [f["dte"].timeline.to_numpy() for f in folds]
        ys = [f["dte"].target.to_numpy() for f in folds]
        ws = [window_weights(f["dte"]) for f in folds]
        out, choices = [], []
        for f in range(len(folds)):
            held = set(recs[f])
            best = None
            for t in TAUS:
                cols = [idx[(a, t)] for a in ALPHAS]
                L = np.zeros((len(ALPHAS), n_regions))
                for g in range(len(folds)):
                    if g == f:
                        continue
                    keep = ~np.isin(recs[g], list(held))
                    e = np.abs(cand[g][cols] - ys[g][None]) * ws[g][None]
                    for r in range(n_regions):
                        L[:, r] += e[:, keep & (regions[g] == r)].sum(1)
                tot = L.min(0).sum()
                if best is None or tot < best[0]:
                    best = (tot, t, L.argmin(0))
            _, t, c = best
            cols = np.array([idx[(a, t)] for a in ALPHAS])
            out.append(cand[f][cols[c[regions[f]]], np.arange(len(ys[f]))].astype(np.float32))
            choices.append(dict(tau=float(t), alphas=[float(ALPHAS[j]) for j in c]))
        return out, choices

    methods["A_pred+T_global"] = region_alpha_global_tau(reg_pred, 4)
    methods["A_elapsed+T_global"] = region_alpha_global_tau(reg_el, 3)

    def describe(name, ch):
        if ch is None:
            return None
        if name in ("A_global", "A_pred", "A_elapsed"):
            return [[float(ALPHAS[j]) for j in c] for c in ch]
        if name in ("T_global", "T_pred"):
            return [[float(TAUS[j]) for j in c] for c in ch]
        if name == "AT_pred":
            return [[pairs[j] for j in c] for c in ch]
        return ch

    summary = {"ref": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ref_b["macro"].items()}}
    print_breakdown("ref = mix_uw_n10 (final540 recipe)", ref_b)
    for name, (preds, ch) in methods.items():
        P = to_protocols(folds, preds)
        for proto in P:
            np.savez(OUT / f"mixcal_{name}_{proto}_preds.npz", *P[proto])
        b = breakdown(d, protocols, P)
        print_breakdown(name, b, ref_b)
        print(f"    choices per fold (unseen 0-4, seen 0-4): {describe(name, ch)}", flush=True)
        summary[name] = dict(dual=float(b["macro"]["wbmae"]), unseen=b["unseen"]["wbmae"], seen=b["seen"]["wbmae"],
                             contrib=b["macro"]["contrib"].tolist(), bmae=float(b["macro"]["bmae"]),
                             mae=float(b["macro"]["mae"]),
                             better_folds=sum(int((np.array(b[p]["folds"]) < np.array(ref_b[p]["folds"])).sum())
                                              for p in ("unseen", "seen")),
                             choices=describe(name, ch))
    (OUT / "mix_calib.json").write_text(json.dumps(summary, indent=2, default=float))
