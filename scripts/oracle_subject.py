"""Oracle diagnostic (NOT a submission model): how much would knowing the sleeper help?

Seen protocol only (5 folds; every held-out night's sleeper has training nights). Per fold, three
columns are appended to the 817 features, computed ONLY from that fold's training recordings:
  oracle_med   = median sleep-onset latency (stream length, s) of the sleeper's OTHER training nights
                 (-1 if none; for a training night its own latency is excluded),
  oracle_n     = number of those nights,
  oracle_rem   = clip(oracle_med - elapsed, 0, 600) (-1 if none): the "remaining time" that median implies.
Member B (dist_hgb, 400 trees, LOFO trees) is retrained with them; compared with the cached B
(ens_dyn_maxmin_spec_eye_B.npz, same folds) alone and mixed with GRU-uw (10 seeds) at an α chosen LOFO.
Reported: seen W-bMAE, per-range contributions, pooled bMAE/MAE, fold-paired differences.

Usage: python scripts/oracle_subject.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ablation_c import ABLATIONS, build_table_ext
from cv_utils import apply_bin_edges, fit_bin_edges
from dist_hgb import PARAMS, VARIANTS, class_stats, decide, to_class
from dual_cv import OUT, STEP, lofo_select, seen_splits
from ensemble_dist import GRID, grid_omega, to_grid
from report_bins import contributions
from stream_utils import RANGE_NAMES, range_index, summarize

B = VARIANTS["B"]
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
ONES = np.ones(len(GRID) - 1)


def oracle_cols(d, onset, rec_sub, tr):
    train_tl = set(d.timeline[tr].unique())
    tab = pd.DataFrame({"tl": onset.index, "sub": rec_sub.to_numpy(), "on": onset.to_numpy()})
    tab = tab[tab.tl.isin(train_tl)]
    by_sub = {s: g for s, g in tab.groupby("sub")}
    med = np.full(len(onset), -1.0)
    cnt = np.zeros(len(onset))
    for i, (tl, s) in enumerate(zip(onset.index, rec_sub.to_numpy())):
        g = by_sub.get(s)
        if g is None:
            continue
        o = g.on[g.tl != tl]
        cnt[i] = len(o)
        if len(o):
            med[i] = o.median()
    pos = d.timeline.map({tl: i for i, tl in enumerate(onset.index)}).to_numpy()
    m, n = med[pos], cnt[pos]
    rem = np.where(n > 0, np.clip(m - d.elapsed_s.to_numpy(), 0, 600), -1.0)
    return np.column_stack([m, n, rem])


def report(name, d, splits, preds, ref=None):
    fold = np.array([summarize(d[te].reset_index(drop=True), p)["wbmae"] for (_, te), p in zip(splits, preds)])
    contrib, _ = contributions(d, splits, preds)
    y = np.concatenate([d.target.to_numpy()[te] for _, te in splits])
    p = np.concatenate([np.asarray(x, float) for x in preds])
    err = np.abs(p - y)
    rng = range_index(y)
    pooled = np.array([err[rng == b].mean() for b in range(4)])
    line = (f"  {name:28s} seen W-bMAE {fold.mean():6.2f} ± {fold.std(ddof=1):.2f} | " +
            "  ".join(f"{r} {c:5.2f}" for r, c in zip(RANGE_NAMES, contrib)) +
            f" | bMAE {pooled.mean():6.2f} MAE {err.mean():6.2f}")
    if ref is not None:
        diff = fold - ref
        line += f" | Δ folds " + " ".join(f"{x:+.2f}" for x in diff) + f" → better {(diff < 0).sum()}/5"
    print(line, flush=True)
    return fold


if __name__ == "__main__":
    wg, sg, drop = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X, names = build_table_ext(wg, sg, drop)
    m = (df.split != "test").to_numpy()
    d, X = df[m].reset_index(drop=True), X[m]
    splits = seen_splits(d)
    onset = d.groupby("timeline").stop.max()
    rec_sub = d.groupby("timeline").subject.first()
    print(f"{X.shape[1]} features + 3 oracle columns; seen folds held-out recordings "
          f"{[int(d[te].timeline.nunique()) for _, te in splits]}", flush=True)

    curves, probas = [], []
    for k, (tr, te) in enumerate(splits):
        t0 = time.perf_counter()
        Xk = np.concatenate([X, oracle_cols(d, onset, rec_sub, tr)], axis=1).astype(np.float32)
        dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
        assert (Xk[te][:, -2] > 0).all(), "a held-out night without training nights of its sleeper"
        reps, omega = class_stats(dtr, B)
        bins = fit_bin_edges(Xk[tr])
        clf = HistGradientBoostingClassifier(max_iter=400, max_features=0.5, early_stopping=False,
                                             random_state=0, **PARAMS)
        clf.fit(apply_bin_edges(Xk[tr], bins), to_class(dtr.target.to_numpy(), B))
        c, pr = [], []
        for n, P in enumerate(clf.staged_predict_proba(apply_bin_edges(Xk[te], bins)), start=1):
            if n % STEP == 0:
                c.append(summarize(dte, decide(P, reps, omega, True, B))["wbmae"])
                pr.append(P.astype(np.float32))
        curves.append(np.array(c))
        probas.append(pr)
        print(f"  fold {k}: in-fold best {STEP * (np.argmin(c) + 1)} trees {min(c):.2f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    P_or, n_iter, _ = lofo_select(curves, probas)
    np.savez(OUT / "oracle_B_seen_proba.npz", *P_or)
    print(f"oracle B trees per fold {n_iter}", flush=True)

    P_b = np.load(OUT / "ens_dyn_maxmin_spec_eye_B.npz", allow_pickle=True)["probas"].item()["seen"]
    z = np.load(OUT / f"gru_uw_n10_seen_proba.npz")
    P_g = [z[f"arr_{k}"] for k in range(len(splits))]

    def own(Ps):
        out = []
        for (tr, _), P in zip(splits, Ps):
            _, om = class_stats(d[tr].reset_index(drop=True), B)
            out.append(decide(P.astype(np.float64), None, om, True, B).astype(np.float32))
        return out

    def mix(Ps):
        cand, curves = [], []
        for (tr, te), P, G in zip(splits, Ps, P_g):
            dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
            Wh = to_grid(P.astype(np.float64), B) * grid_omega(dtr)
            Wh /= Wh.sum(1, keepdims=True)
            _, ob = class_stats(dtr, B)
            q = G.astype(np.float64) * ob
            q = to_grid(q / q.sum(1, keepdims=True), B)
            c = [decide(a * q + (1 - a) * Wh, None, ONES, True, GRID).astype(np.float32) for a in ALPHAS]
            cand.append(c)
            curves.append([summarize(dte, x)["wbmae"] for x in c])
        C = np.array(curves)
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
        return [cand[k][j] for k, j in enumerate(sel)], [float(ALPHAS[j]) for j in sel]

    print("== seen protocol (5 folds)")
    f_b = report("B (817 feats)", d, splits, own(P_b))
    report("B + oracle (820 feats)", d, splits, own(P_or), f_b)
    mb, ab = mix(P_b)
    f_mb = report(f"B + GRU, α {ab}", d, splits, mb)
    mo, ao = mix(P_or)
    report(f"B_oracle + GRU, α {ao}", d, splits, mo, f_mb)
    # how predictive is the oracle column alone: rem vs target, W-bMAE of using rem directly
    rem_preds = [np.clip(oracle_cols(d, onset, rec_sub, tr)[te][:, 2], 0, 600).astype(np.float32) for tr, te in splits]
    report("oracle_rem alone (no EEG)", d, splits, rem_preds)
