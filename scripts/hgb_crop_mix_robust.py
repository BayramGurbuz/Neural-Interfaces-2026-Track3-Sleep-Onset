"""Stream-start robustness of the MIX with a crop-augmented HGB member (follow-up of hgb_crop_precheck.py, which gave the
HGB-only columns). Train+val models, the 40 public-test recordings, as stream_start_test.py.

Sequence side = the bundle given by --bundle (default mix3crop: GRU / TCN / GRU-emb all crop-augmented). HGB side = the
single B member (817 features, n_iter 350 as in the pre-check) trained (a) on the original rows, (b) on the original rows
+ one start-cropped copy of every training recording (hgb_crop_precheck.cropped_copies, seed 0 -> the same copies).
Mix = alpha * seq + (1 - alpha) * B on the grid, ungridded decision; alpha = the bundle's alpha and 0.6.
For k = 1, 3, 6, 12, 24, 60, random: Δ = truncated stream - full-history predictions on the same windows, for the mix,
the B member alone and the sequence side alone. The 4-member HGB(c) columns of the same bundle are in
OUT/stream_start_test_crop.json.

Usage: python scripts/hgb_crop_mix_robust.py [--bundle=mix3crop]   -> OUT/hgb_crop_mix_robust.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

if not any(a.startswith("--bundle=") for a in sys.argv):
    sys.argv.append("--bundle=mix3crop")
import stream_start_test as sst  # noqa: E402  (loads the bundle and the test tables at import)
import hgb_crop_precheck as pc  # noqa: E402
from cv_utils import apply_bin_edges, fit_bin_edges  # noqa: E402
from dist_hgb import PARAMS, decide, to_class  # noqa: E402
from dual_cv import OUT  # noqa: E402
from ensemble_dist import GRID, to_grid  # noqa: E402

N_ITER = 350
KS = {"k1": 1, "k3": 3, "k6": 6, "k12": 12, "k24": 24, "k60": 60, "random": None}
ALPHAS = sorted({float(sst.alpha), 0.6})

if __name__ == "__main__":
    t_all = time.perf_counter()
    m = (pc.df.split != "test").to_numpy()
    d = pc.df[m].reset_index(drop=True)
    X = pc.X817[m]
    recs_tr = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    c_rows, c_X, ks_tr = pc.cropped_copies(recs_tr, np.random.default_rng(0))
    d_ext = pd.concat([d, d.iloc[c_rows]], ignore_index=True)
    X_ext = np.concatenate([X, c_X])
    omega = np.asarray(sst.b["grid_omega"])
    ones = np.ones(len(GRID) - 1)
    recs = sst.recs
    onset = [sst.onset[i] for i in range(len(recs))]
    rng = np.random.default_rng(0)
    ks_all = {}
    for name, k in KS.items():
        kk = [k if k is not None else int(rng.integers(0, max(int((o - 60) // 5), 0) + 1)) for o in onset]
        ks_all[name] = [min(q, len(r) - 1) for q, r in zip(kk, recs)]
    # sequence side and re-streamed features once per (recording, k); k = 0 first
    t0 = time.perf_counter()
    seq = {"k0": [sst.predict_from(r, 0) for r in recs]}
    for name, kk in ks_all.items():
        seq[name] = [sst.predict_from(r, q) for r, q in zip(recs, kk)]
    print(f"bundle {sst.BUNDLE}: sequence side + re-streamed features for {len(seq)} settings in {time.perf_counter() - t0:.0f} s", flush=True)
    res = {"bundle": sst.BUNDLE, "n_iter": N_ITER, "alphas": ALPHAS, "members": {}}
    for mname, (Xtr, ytr) in {"B_plain": (X, d.target.to_numpy()), "B_crop": (X_ext, d_ext.target.to_numpy())}.items():
        t0 = time.perf_counter()
        bins = fit_bin_edges(Xtr)
        clf = HistGradientBoostingClassifier(max_iter=N_ITER, max_features=0.5, early_stopping=False, random_state=0, **PARAMS)
        clf.fit(apply_bin_edges(Xtr, bins), to_class(ytr, pc.B))

        def dists(f):
            Wh = to_grid(clf.predict_proba(apply_bin_edges(f["H"], bins)), pc.B) * omega
            return Wh / Wh.sum(1, keepdims=True), f["qs"]

        def decisions(f):
            Wh, qs = dists(f)
            out = {"hgb": decide(Wh, None, ones, True, GRID), "seq": decide(qs, None, ones, True, GRID)}
            for a in ALPHAS:
                out[f"mix_a{a}"] = decide(a * qs + (1 - a) * Wh, None, ones, True, GRID)
            return out

        full = [decisions(f) for f in seq["k0"]]
        out = {"k0": {c: sst.score(recs, [f[c] for f in full])[0] for c in full[0]}}
        for name, kk in ks_all.items():
            tr = [decisions(f) for f in seq[name]]
            rows_list = [r[q:] for r, q in zip(recs, kk)]
            out[name] = {}
            for c in full[0]:
                sf, cf = sst.score(rows_list, [f[c][q:] for f, q in zip(full, kk)])
                st, ct = sst.score(rows_list, [f[c] for f in tr])
                out[name][c] = dict(full=sf, truncated=st, delta=st - sf, d_contrib=(np.array(ct) - np.array(cf)).tolist())
        res["members"][mname] = out
        print(f"{mname} (fit + eval {time.perf_counter() - t0:.0f} s): k0 " + " ".join(f"{c} {v:.2f}" for c, v in out["k0"].items()), flush=True)
        for c in full[0]:
            print(f"  {c:9s} Δ " + " ".join(f"{n} {out[n][c]['delta']:+.2f}" for n in KS), flush=True)
    res["time_s"] = time.perf_counter() - t_all
    (OUT / "hgb_crop_mix_robust.json").write_text(json.dumps(res, indent=2))
    print(f"saved {OUT / 'hgb_crop_mix_robust.json'} ({res['time_s']:.0f} s)")
