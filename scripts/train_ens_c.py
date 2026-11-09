"""Ensemble (c): B + B_h1 + B_h2 + B2 distribution classifiers on dyn+maxmin+spec+eye features
(817), trained on all of train+val, averaged on the common grid, embedded calibration.

- per member: params / class edges from ensemble_dist.MEMBERS; n_iter = median of its 10 dual-CV
  LOFO choices (cached in ens_dyn_maxmin_spec_eye_<member>.npz), rounded to 25.
- decision: mean grid distribution -> continuous weighted median with grid omega from train+val.
- calibration map: fit_monotone_map on the unseen-protocol OOF predictions of ensemble (c).
Test is scored once, as a final check only.

Usage: python scripts/train_ens_c.py   (writes ~/neuralbench/save/ens_c/)
"""
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ablation_c import ABLATIONS, build_table_ext
from calibrate_oof import fit_monotone_map
from cv_utils import apply_bin_edges, fit_bin_edges
from dist_hgb import decide, to_class
from dual_cv import OUT as CV_OUT, unseen_splits
from ensemble_dist import GRID, MEMBERS, grid_omega, to_grid
from stream_utils import RANGE_NAMES, summarize, window_weights
from train_hgb import SAVE

FEAT_SET = "dyn_maxmin_spec_eye"
OUT = SAVE / "ens_c"
OUT.mkdir(parents=True, exist_ok=True)

wg, sg, _ = ABLATIONS[FEAT_SET]
df, X, names = build_table_ext(wg, sg)
m_tv = (df.split != "test").to_numpy()
d_tv, d_te = df[m_tv].reset_index(drop=True), df[~m_tv].reset_index(drop=True)

z = np.load(CV_OUT / f"ens_{FEAT_SET}_(c)_B-B_h1-B_h2-B2@grid_unseen_preds.npz")
oof = np.empty(len(d_tv))
for k, (_, te) in enumerate(unseen_splits(d_tv)):
    oof[te] = z[f"arr_{k}"]
cal_x, cal_y = fit_monotone_map(oof, d_tv.target.to_numpy(), window_weights(d_tv), 20)

bins = fit_bin_edges(X[m_tv])
Xtr, Xte = apply_bin_edges(X[m_tv], bins), apply_bin_edges(X[~m_tv], bins)
models, info = [], {}
t_all = time.perf_counter()
for name, (edges, params, _) in MEMBERS.items():
    it = np.load(CV_OUT / f"ens_{FEAT_SET}_{name}.npz", allow_pickle=True)["n_iter"].item()
    n_iter = int(25 * round(np.median(it["unseen"] + it["seen"]) / 25))
    t0 = time.perf_counter()
    clf = HistGradientBoostingClassifier(max_iter=n_iter, max_features=0.5, early_stopping=False,
                                         random_state=0, **params)
    clf.fit(Xtr, to_class(d_tv.target.to_numpy(), edges))
    info[name] = dict(n_iter=n_iter, fit_s=time.perf_counter() - t0)
    print(f"{name}: {n_iter} iters, fit {info[name]['fit_s']:.0f} s", flush=True)
    models.append((clf, edges))

omega = grid_omega(d_tv)
P = np.mean([to_grid(clf.predict_proba(Xte), e) for clf, e in models], axis=0)
raw = decide(P, None, omega, True, GRID)
cal = np.clip(np.interp(raw, cal_x, cal_y), 0, 600)
scores = {k: summarize(d_te, p) for k, p in (("plain", raw), ("calib", cal))}
for k, v in scores.items():
    print(f"test {k:6s} {v['wbmae']:.4f}  " + " ".join(f"{r} {v[r]:.1f}" for r in RANGE_NAMES))
print(f"total fit {time.perf_counter() - t_all:.0f} s, {X.shape[1]} features")

joblib.dump(dict(kind="ens_grid", models=[c for c, _ in models], member_edges=[e for _, e in models],
                 grid=GRID, grid_omega=omega, feature_names=names, bin_edges=bins,
                 calib_x=cal_x, calib_y=cal_y, groups=sg, window_groups=wg), OUT / "model.joblib")
(OUT / "results.json").write_text(json.dumps(dict(
    feat_set=FEAT_SET, members=info, calib_x=cal_x.tolist(), calib_y=cal_y.tolist(), test=scores), indent=2))
