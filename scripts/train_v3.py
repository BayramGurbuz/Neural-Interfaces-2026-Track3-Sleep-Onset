"""v3: distribution model B (dist_hgb, continuous decision) + "dyn" features, trained on all of
train+val, with embedded monotone calibration.

- features: base 321 + dyn group (ablation_c.build_table_ext((), ("dyn",))) = 517.
- classifier: dist_hgb.PARAMS, class edges VARIANTS["B"], N_ITER = median of the 10 dual-CV
  LOFO choices (unseen 225-275, seen 300-375) -> 300.
- reps / omega: class_stats on train+val.
- calibration map: fit_monotone_map on the unseen-protocol OOF predictions of the B+dyn ablation
  (every train+val recording held out exactly once).
- inference: P -> continuous weighted median -> calibration (np.interp) -> clip [0, 600].
Test is scored once, as a final check only.

Usage: python scripts/train_v3.py   (writes ~/neuralbench/save/hgb_v3/)
"""
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ablation_c import build_table_ext
from calibrate_oof import fit_monotone_map
from cv_utils import apply_bin_edges, fit_bin_edges
from dist_hgb import PARAMS, VARIANTS, class_stats, decide, to_class
from dual_cv import OUT as CV_OUT, unseen_splits
from stream_utils import RANGE_NAMES, summarize, window_weights
from train_hgb import SAVE

OUT = SAVE / "hgb_v3"
OUT.mkdir(parents=True, exist_ok=True)
N_ITER = 300
EDGES = VARIANTS["B"]
GROUPS = ("dyn",)

df, X, names = build_table_ext((), GROUPS)
m_tv = (df.split != "test").to_numpy()
d_tv, d_te = df[m_tv].reset_index(drop=True), df[~m_tv].reset_index(drop=True)

# calibration from the unseen OOF predictions
z = np.load(CV_OUT / "abl_B_dyn_unseen_preds.npz")
splits = unseen_splits(d_tv)
oof = np.empty(len(d_tv))
for k, (_, te) in enumerate(splits):
    oof[te] = z[f"arr_{k}"]
cal_x, cal_y = fit_monotone_map(oof, d_tv.target.to_numpy(), window_weights(d_tv), 20)

t0 = time.perf_counter()
reps, omega = class_stats(d_tv, EDGES)
bins = fit_bin_edges(X[m_tv])
clf = HistGradientBoostingClassifier(max_iter=N_ITER, max_features=0.5, early_stopping=False,
                                     random_state=0, **PARAMS)
clf.fit(apply_bin_edges(X[m_tv], bins), to_class(d_tv.target.to_numpy(), EDGES))
fit_s = time.perf_counter() - t0

P = clf.predict_proba(apply_bin_edges(X[~m_tv], bins))
raw = decide(P, reps, omega, True, EDGES)
cal = np.clip(np.interp(raw, cal_x, cal_y), 0, 600)
scores = {k: summarize(d_te, p) for k, p in (("plain", raw), ("calib", cal))}
for k, v in scores.items():
    print(f"test {k:6s} {v['wbmae']:.4f}  " + " ".join(f"{r} {v[r]:.1f}" for r in RANGE_NAMES))
print(f"fit {fit_s:.0f} s, {N_ITER} iters x {len(reps)} classes, {X.shape[1]} features")
print("calibration map:", ", ".join(f"{a:.1f}->{b:.1f}" for a, b in zip(cal_x, cal_y)))

joblib.dump(dict(model=clf, feature_names=names, bin_edges=bins, calib_x=cal_x, calib_y=cal_y,
                 kind="dist", class_edges=EDGES, reps=reps, omega=omega, groups=GROUPS),
            OUT / "model.joblib")
(OUT / "results.json").write_text(json.dumps(dict(
    params=PARAMS, n_iter=N_ITER, groups=GROUPS, fit_s=fit_s, reps=reps.tolist(), omega=omega.tolist(),
    calib_x=cal_x.tolist(), calib_y=cal_y.tolist(), test=scores), indent=2))
