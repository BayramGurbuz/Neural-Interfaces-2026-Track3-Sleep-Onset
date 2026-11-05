"""HistGradientBoostingRegressor(loss="absolute_error") on causal features, W-bMAE sample weights.

Steps:
1. Build the full causal feature vectors by streaming each recording's cached base features
   through CausalFeatureStream (same code the submission will run).
2. Pick the number of trees on a subject-grouped inner hold-out of train (val is not used for
   any choice), then refit on all of train.
3. Report val (unseen subjects), test (seen subjects), their mean and test per-range MAE;
   permutation importance on val (top 15); ablation without the *_rel features.

Usage: python scripts/train_hgb.py   (writes ~/neuralbench/save/hgb/)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from causal_features import CausalFeatureStream, feature_names
from stream_utils import RANGE_NAMES, load_window_table, summarize, window_weights

SAVE = Path.home() / "neuralbench" / "save"
OUT = SAVE / "hgb"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
PARAMS = dict(loss="absolute_error", learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, max_features=0.5, early_stopping=False, random_state=SEED)
MAX_ITER = 1500
STAGE = 25


def build_table():
    z = np.load(SAVE / "features_base.npz")
    names = feature_names(list(z["ch_names"]))
    base = pd.DataFrame({"timeline": z["timeline"], "stop": z["stop"], "split": z["split"]})
    order = np.lexsort((base.stop.to_numpy(), base.timeline.to_numpy()))
    feats = z["feats"].astype(np.float64)
    full = np.empty((len(base), len(names)), dtype=np.float32)
    stream = CausalFeatureStream(step_s=5.0)
    prev = None
    for i in order:
        if base.timeline.iat[i] != prev:
            stream.reset()
            prev = base.timeline.iat[i]
        full[i] = stream.update(feats[i])
    windows = load_window_table()  # targets, subjects, elapsed (reference)
    df = base.merge(windows, on=["split", "timeline", "stop"], how="left", validate="1:1")
    assert df.target.notna().all()
    assert np.allclose(full[:, -1], df.elapsed_s.to_numpy()), "elapsed mismatch"
    df = df.reset_index(drop=True)
    return df, full, names


def norm_weights(d):
    w = window_weights(d)
    return w / w.mean()


def fit_select(Xtr, dtr, cols, rng):
    """Choose n_iter on a subject-grouped inner hold-out of train, then refit on all train."""
    subjects = dtr.subject.unique()
    hold = set(rng.choice(subjects, size=int(0.15 * len(subjects)), replace=False))
    m_in = ~dtr.subject.isin(hold).to_numpy()
    model = HistGradientBoostingRegressor(max_iter=MAX_ITER, **PARAMS)
    model.fit(Xtr[m_in][:, cols], dtr.target[m_in], sample_weight=norm_weights(dtr[m_in]))
    d_ho = dtr[~m_in]
    curve = []
    for k, p in enumerate(model.staged_predict(Xtr[~m_in][:, cols]), start=1):
        if k % STAGE == 0:
            curve.append((k, summarize(d_ho, p)["wbmae"]))
    n_best, s_best = min(curve, key=lambda t: t[1])
    model = HistGradientBoostingRegressor(max_iter=n_best, **PARAMS)
    model.fit(Xtr[:, cols], dtr.target, sample_weight=norm_weights(dtr))
    return model, n_best, s_best, curve


def evaluate(model, X, d, cols):
    return summarize(d, model.predict(X[:, cols]))


def permutation_importance_wbmae(model, X, d, cols, names, n_repeats, rng):
    base = evaluate(model, X, d, cols)["wbmae"]
    Xc = X[:, cols].copy()
    imp = []
    for j in range(len(cols)):
        orig = Xc[:, j].copy()
        deltas = []
        for _ in range(n_repeats):
            Xc[:, j] = rng.permutation(orig)
            deltas.append(summarize(d, model.predict(Xc))["wbmae"] - base)
        Xc[:, j] = orig
        imp.append((names[cols[j]], float(np.mean(deltas)), float(np.std(deltas))))
    return sorted(imp, key=lambda t: -t[1])


if __name__ == "__main__":
    t0 = time.perf_counter()
    df, X, names = build_table()
    print(f"features: {X.shape} built in {time.perf_counter() - t0:.0f} s", flush=True)
    sp = {s: (df.split == s).to_numpy() for s in ("train", "val", "test")}
    d = {s: df[m].reset_index(drop=True) for s, m in sp.items()}
    Xs = {s: X[m] for s, m in sp.items()}

    results = {}
    variants = {
        "all": list(range(len(names))),
        "no_rel": [i for i, n in enumerate(names) if not n.endswith("_rel")],
    }
    models = {}
    for vname, cols in variants.items():
        t1 = time.perf_counter()
        model, n_best, s_ho, curve = fit_select(Xs["train"], d["train"], cols,
                                               np.random.default_rng(SEED))
        r = {s: evaluate(model, Xs[s], d[s], cols) for s in ("train", "val", "test")}
        results[vname] = dict(n_features=len(cols), n_iter=n_best, inner_holdout_wbmae=s_ho,
                              fit_time_s=time.perf_counter() - t1, curve=curve,
                              **{s: r[s] for s in r},
                              mean_val_test=(r["val"]["wbmae"] + r["test"]["wbmae"]) / 2)
        models[vname] = model
        print(f"[{vname}] {len(cols)} feats, n_iter={n_best} (inner {s_ho:.2f}) "
              f"val {r['val']['wbmae']:.2f} test {r['test']['wbmae']:.2f} "
              f"train {r['train']['wbmae']:.2f}  ({results[vname]['fit_time_s']:.0f} s)", flush=True)
        print("   test per range:", {k: round(r["test"][k], 2) for k in RANGE_NAMES}, flush=True)

    joblib.dump(dict(model=models["all"], feature_names=names, params=PARAMS),
                OUT / "hgb_all.joblib")
    t2 = time.perf_counter()
    imp = permutation_importance_wbmae(models["all"], Xs["val"], d["val"], variants["all"], names,
                                       n_repeats=3, rng=np.random.default_rng(SEED))
    results["permutation_importance_val_top30"] = imp[:30]
    print(f"permutation importance ({time.perf_counter() - t2:.0f} s), top 15:")
    for n, m, s in imp[:15]:
        print(f"  {n:28s} +{m:.3f} ± {s:.3f}")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT)
