"""v2: c5 HGB trained on all train+val windows, pre-binned features, embedded monotone calibration.

- params = grid config c5; n_iter = argmin of the 5-fold mean CV curve (cv_hgb_grid.py).
- bin edges fit on train+val (constants at inference).
- calibration map = fit_monotone_map on all c5 OOF predictions (20 bins).
- inference: pred -> calibration (np.interp) -> causal smoothing (a=0.3) -> clip [0, 600].
Test (seen subjects) is scored once, as a final check only.

Usage: python scripts/train_v2.py   (writes ~/neuralbench/save/hgb_v2/)
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from calibrate_oof import fit_monotone_map
from cv_utils import apply_bin_edges, causal_smooth, fit_bin_edges
from stream_utils import RANGE_NAMES, summarize, window_weights
from train_hgb import build_table, norm_weights

SAVE = Path.home() / "neuralbench" / "save"
OUT = SAVE / "hgb_v2"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = "c5"
SMOOTH_A = 0.3

grid = json.loads((SAVE / "cv" / "grid" / "summary.json").read_text())[CONFIG]
params, n_iter = grid["params"], grid["n_iter_meancurve"]
oof = pd.read_parquet(SAVE / "cv" / "grid" / f"{CONFIG}_oof.parquet").reset_index(drop=True)
cal_x, cal_y = fit_monotone_map(oof.pred.to_numpy(), oof.target.to_numpy(), window_weights(oof), 20)

df, X, names = build_table()
m_tv = (df.split != "test").to_numpy()
d_tv, d_te = df[m_tv].reset_index(drop=True), df[~m_tv].reset_index(drop=True)
t0 = time.perf_counter()
edges = fit_bin_edges(X[m_tv])
model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=n_iter, max_features=0.5,
                                      early_stopping=False, random_state=0, **params)
model.fit(apply_bin_edges(X[m_tv], edges), d_tv.target, sample_weight=norm_weights(d_tv))
fit_s = time.perf_counter() - t0

raw = model.predict(apply_bin_edges(X[~m_tv], edges))
cal = np.interp(raw, cal_x, cal_y)
final = np.clip(causal_smooth(d_te, cal, SMOOTH_A), 0, 600)
scores = {name: summarize(d_te, p) for name, p in
          (("raw", raw), ("calib", cal), ("calib+smooth", final))}
for k, v in scores.items():
    print(f"test {k:13s} {v['wbmae']:.4f}  " + " ".join(f"{r} {v[r]:.1f}" for r in RANGE_NAMES))
print(f"fit {fit_s:.0f} s, {n_iter} trees, params {params}")
print("calibration map:", ", ".join(f"{a:.1f}->{b:.1f}" for a, b in zip(cal_x, cal_y)))

joblib.dump(dict(model=model, feature_names=names, bin_edges=edges,
                 calib_x=cal_x, calib_y=cal_y), OUT / "model.joblib")
(OUT / "results.json").write_text(json.dumps(dict(
    config=CONFIG, params=params, n_iter=n_iter, smooth_a=SMOOTH_A, fit_s=fit_s,
    calib_x=cal_x.tolist(), calib_y=cal_y.tolist(), test=scores), indent=2))
