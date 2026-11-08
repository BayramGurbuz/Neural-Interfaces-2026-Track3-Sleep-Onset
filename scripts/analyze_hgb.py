"""HGB v1 analysis: near-onset behaviour, causal smoothing, 5-fold subject-grouped CV.

1. Near onset (true target 0-60 s) on val/test: mean prediction vs truth per 10 s bin, and the
   per-recording split of the error into a systematic part (bias) and a jumpy part (std).
2. Smoothing s_t = a*max(s_{t-1}-dt,0) + (1-a)*p_t on existing predictions; a picked on the
   train inner hold-out (same subjects / 50 trees as train_hgb.py), then val/test reported.
3. 5-fold subject-grouped CV over train+val: out-of-fold predictions (saved for later
   comparisons), smoothing compared as mean ± std over folds.

Usage: python scripts/analyze_hgb.py   (writes ~/neuralbench/save/cv/)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cv_utils import N_FOLDS, causal_smooth, make_folds
from stream_utils import RANGE_NAMES, summarize
from train_hgb import PARAMS, SEED, build_table, fit_select, norm_weights

SAVE = Path.home() / "neuralbench" / "save"
OUT = SAVE / "cv"
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = (0.0, 0.3, 0.5, 0.7, 0.9)
CV_MAX_ITER = 300


def near_onset(d, p, name):
    m = (d.target < 60).to_numpy()
    x = d[m].assign(pred=p[m], err=p[m] - d.target[m].to_numpy())
    x["bin"] = (x.target // 10 * 10).astype(int)
    tab = x.groupby("bin").agg(n=("err", "size"), true_mean=("target", "mean"),
                               pred_mean=("pred", "mean"), pred_median=("pred", "median"),
                               bias=("err", "mean"), mae=("err", lambda e: e.abs().mean()))
    rec = x.groupby("timeline").agg(bias=("err", "mean"), std=("err", "std"),
                                    jump=("pred", lambda q: q.diff().abs().mean()))
    print(f"\n[{name}] windows with true target < 60 s, per 10 s bin:")
    print(tab.round(1).to_string())
    print(f"  per recording: mean bias {rec.bias.mean():+.1f} s, mean |bias| {rec.bias.abs().mean():.1f} s, "
          f"mean within-rec std {rec['std'].mean():.1f} s, mean |p_t - p_t-1| {rec.jump.mean():.1f} s "
          f"(truth moves 5 s/window); share of recs over-predicting {np.mean(rec.bias > 0):.0%}")
    return dict(table=tab.reset_index().to_dict("records"), rec_bias_mean=rec.bias.mean(),
                rec_abs_bias_mean=rec.bias.abs().mean(), rec_std_mean=rec["std"].mean(),
                jump_mean=rec.jump.mean(), share_over=float(np.mean(rec.bias > 0)))


def smooth_scores(d, p):
    return {a: summarize(d, causal_smooth(d, p, a)) for a in ALPHAS}


def fmt(r):
    return f"{r['wbmae']:.2f} | " + " ".join(f"{r[k]:.1f}" for k in RANGE_NAMES)


if __name__ == "__main__":
    t0 = time.perf_counter()
    df, X, names = build_table()
    cols = list(range(len(names)))
    sp = {s: (df.split == s).to_numpy() for s in ("train", "val", "test")}
    d = {s: df[m].reset_index(drop=True) for s, m in sp.items()}
    Xs = {s: X[m] for s, m in sp.items()}
    res = json.loads((SAVE / "hgb" / "results.json").read_text())["all"]
    full = joblib.load(SAVE / "hgb" / "hgb_all.joblib")["model"]
    p = {s: full.predict(Xs[s]) for s in ("val", "test")}
    report = {}

    # ---- 1. near-onset analysis
    report["near_onset"] = {s: near_onset(d[s], p[s], s) for s in ("val", "test")}

    # ---- 2. smoothing, a chosen on the train inner hold-out
    rng = np.random.default_rng(SEED)  # same draw as train_hgb.fit_select -> same hold-out
    subjects = d["train"].subject.unique()
    hold = set(rng.choice(subjects, size=int(0.15 * len(subjects)), replace=False))
    m_in = ~d["train"].subject.isin(hold).to_numpy()
    inner = HistGradientBoostingRegressor(max_iter=res["n_iter"], **PARAMS)
    inner.fit(Xs["train"][m_in], d["train"].target[m_in], sample_weight=norm_weights(d["train"][m_in]))
    d_ho = d["train"][~m_in].reset_index(drop=True)
    sm_ho = smooth_scores(d_ho, inner.predict(Xs["train"][~m_in]))
    a_best = min(ALPHAS, key=lambda a: sm_ho[a]["wbmae"])
    sm = {s: smooth_scores(d[s], p[s]) for s in ("val", "test")}
    print("\nSmoothing (W-bMAE | " + " ".join(RANGE_NAMES) + "):")
    for a in ALPHAS:
        print(f"  a={a:.1f}  inner {fmt(sm_ho[a])}   val {fmt(sm['val'][a])}   test {fmt(sm['test'][a])}")
    print(f"  -> a picked on inner hold-out: {a_best}")
    report["smoothing_inner"] = dict(a_best=a_best, inner=sm_ho, val=sm["val"], test=sm["test"])

    # ---- 3. 5-fold subject-grouped CV over train+val
    m_tv = sp["train"] | sp["val"]
    dtv = df[m_tv].reset_index(drop=True)
    Xtv = X[m_tv]
    fold = make_folds(dtv)
    oof = np.full(len(dtv), np.nan)
    fold_info = []
    for k in range(N_FOLDS):
        t1 = time.perf_counter()
        tr, te = fold != k, fold == k
        model, n_best, s_ho, _ = fit_select(Xtv[tr], dtv[tr].reset_index(drop=True), cols,
                                            np.random.default_rng(SEED + k), max_iter=CV_MAX_ITER)
        oof[te] = model.predict(Xtv[te])
        fold_info.append(dict(fold=k, n_rec=int(dtv[te].timeline.nunique()),
                              n_subj=int(dtv[te].subject.nunique()), n_iter=n_best,
                              time_s=time.perf_counter() - t1))
        print(f"fold {k}: {fold_info[-1]}", flush=True)
    dtv.assign(fold=fold, pred=oof).to_parquet(OUT / "hgb_v1_oof.parquet")
    cv = {a: [summarize(dtv[fold == k], causal_smooth(dtv[fold == k], oof[fold == k], a))
              for k in range(N_FOLDS)] for a in ALPHAS}
    print("\n5-fold CV (train+val, subject-grouped), W-bMAE mean ± std over folds:")
    cv_summary = {}
    for a in ALPHAS:
        w = np.array([r["wbmae"] for r in cv[a]])
        rngs = {k: float(np.mean([r[k] for r in cv[a]])) for k in RANGE_NAMES}
        cv_summary[a] = dict(mean=float(w.mean()), std=float(w.std(ddof=1)), folds=w.tolist(), **rngs)
        print(f"  a={a:.1f}  {w.mean():.2f} ± {w.std(ddof=1):.2f}   folds {np.round(w, 2).tolist()}   "
              + " ".join(f"{rngs[k]:.1f}" for k in RANGE_NAMES))
    report["cv"] = dict(folds=fold_info, smoothing=cv_summary)
    (OUT / "hgb_v1_analysis.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"done in {time.perf_counter() - t0:.0f} s")
