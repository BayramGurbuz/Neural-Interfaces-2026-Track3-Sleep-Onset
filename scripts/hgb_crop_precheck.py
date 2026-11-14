"""HGB stream-start augmentation, B-member pre-check (14 Nov; user: only the B member before deciding on 4 members).

Training table = original rows + one start-cropped copy of every training recording: the 62 window features are
re-streamed through CausalFeatureStream from window k (elapsed, EMAs, dynamics, running extrema, first-2-min
differences restart there; targets unchanged). k per recording (seed 0): 70 % uniform 1..12 windows (5-60 s), 30 %
uniform 13..len-12 (up to onset - 60 s; small crop if the recording is too short).
 1. Dual CV of the B member (817 features, same settings as b2_ablation: VARIANTS["B"], PARAMS, 400 trees, LOFO) with the
    cropped copies added to each fold's training rows only; held-out rows are the original streams (k = 0).
    Reference: B on the 817 features without augmentation (ens_dyn_maxmin_spec_eye_B, 43.20).
 2. Robustness (HGB-only, the 40 public-test recordings, train+val models): B trained on train+val with / without the
    cropped copies (n_iter = median of the LOFO choices), evaluated with the first k windows dropped (k = 1, 3, 6, 12,
    24, 60, random) against its own full-history predictions on the same windows (as stream_start_test).
Usage: python scripts/hgb_crop_precheck.py   -> OUT/hgb_crop_precheck.json, OUT/b2_crop_B_{unseen,seen}_preds.npz
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ablation_c import ABLATIONS, build_table_ext
from b2_ablation import REF
from causal_features import CausalFeatureStream, core_index
from cv_utils import apply_bin_edges, fit_bin_edges
from decision_layer import RANGE_NAMES, WEIGHTS, official_rec_scores, per_recording_scores
from dist_hgb import PARAMS, VARIANTS, class_stats, decide, run_clf, to_class
from dual_cv import OUT, seen_splits, unseen_splits
from range_breakdown import breakdown, load_preds
from train_hgb import SAVE

B = VARIANTS["B"]
KS = {"k1": 1, "k3": 3, "k6": 6, "k12": 12, "k24": 24, "k60": 60, "random": None}

wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, X817, names = build_table_ext(wg, sg)
zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
F = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)
assert (zb["timeline"] == df.timeline.to_numpy()).all()
stream = CausalFeatureStream(5.0, tuple(sg), core_index(range(4)))
recs_all = [g.sort_values("stop").index.to_numpy() for _, g in df.groupby("timeline", sort=False)]


def restream(rows):
    stream.reset()
    return np.stack([stream.update(F[i].astype(np.float64)) for i in rows]).astype(np.float32)


def crop_k(n, rng, p_small=0.7):
    if rng.random() < p_small or n - 12 < 13:
        return int(rng.integers(1, 13))
    return int(rng.integers(13, n - 12 + 1))


def cropped_copies(recs, rng):
    """-> rows (original indices of the kept windows), features re-streamed from the crop, k per recording."""
    rows, feats, ks = [], [], []
    for r in recs:
        k = min(crop_k(len(r), rng), len(r) - 1)
        rows.append(r[k:])
        feats.append(restream(r[k:]))
        ks.append(k)
    return np.concatenate(rows), np.concatenate(feats), ks


def score_rows(d_rows, pred):
    tot = float(official_rec_scores(d_rows, pred).mean())
    mae = per_recording_scores(d_rows, pred)[RANGE_NAMES].to_numpy()
    w = np.where(np.isnan(mae), 0.0, WEIGHTS)
    return tot, (np.nan_to_num(mae) * w / w.sum(1, keepdims=True)).mean(0).tolist()


if __name__ == "__main__":
    t_all = time.perf_counter()
    m = (df.split != "test").to_numpy()
    d = df[m].reset_index(drop=True)
    X = X817[m]
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    assert np.abs(restream(recs[0]) - X[recs[0]]).max() == 0.0, "re-streamed features differ from the cached table"
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    c_rows, c_X, ks = cropped_copies(recs, rng)
    print(f"cropped copies: {len(c_rows)} windows from {len(recs)} recordings (k median {int(np.median(ks))}, "
          f"small {sum(k <= 12 for k in ks)}, wide {sum(k > 12 for k in ks)}) in {time.perf_counter() - t0:.0f} s", flush=True)
    d_ext = pd.concat([d, d.iloc[c_rows]], ignore_index=True)
    X_ext = np.concatenate([X, c_X])
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    ref = breakdown(d, protocols, load_preds(REF, protocols))
    preds, n_iter = {}, {}
    for proto, splits in protocols.items():
        t0 = time.perf_counter()
        tl_c = d.timeline.to_numpy()[c_rows]
        ext = []
        for tr, te in splits:
            tr_recs = set(d.timeline.to_numpy()[tr])
            ext.append((np.concatenate([tr, np.isin(tl_c, list(tr_recs))]), np.concatenate([te, np.zeros(len(c_rows), bool)])))
        r = run_clf(X_ext, d_ext, ext, B, PARAMS, 400)["dist_continuous"]
        preds[proto], n_iter[proto] = r[0], r[1]
        np.savez(OUT / f"b2_crop_B_{proto}_preds.npz", *preds[proto])
        print(f"  {proto}: n_iter {r[1]} ({time.perf_counter() - t0:.0f} s)", flush=True)
    b = breakdown(d, protocols, preds)
    du = np.array(b["unseen"]["folds"]) - np.array(ref["unseen"]["folds"])
    ds = np.array(b["seen"]["folds"]) - np.array(ref["seen"]["folds"])
    cv = dict(unseen=float(b["unseen"]["wbmae"]), seen=float(b["seen"]["wbmae"]), macro=float(b["macro"]["wbmae"]),
              delta=float(b["macro"]["wbmae"] - ref["macro"]["wbmae"]), better=int((du < 0).sum() + (ds < 0).sum()),
              d_unseen=du.tolist(), d_seen=ds.tolist(),
              d_contrib=(np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])).tolist(), n_iter=n_iter, ks=ks)
    print(f"B + crop copies: dual {cv['macro']:.2f} Δ vs B(817) {cv['delta']:+.2f} better {cv['better']}/10 | Δ contrib "
          + " ".join(f"{x:+.2f}" for x in cv["d_contrib"]) + f" | folds Δ unseen {np.round(du, 2).tolist()} seen {np.round(ds, 2).tolist()}", flush=True)

    # 2. robustness, HGB-only, train+val models with / without crops, on the test recordings
    mt = ~m
    d_te = df[mt].reset_index(drop=True)
    te_recs = [g.sort_values("stop").index.to_numpy() for _, g in d_te.groupby("timeline", sort=False)]
    Fte_rows = np.flatnonzero(mt)
    it = int(25 * round(np.median(n_iter["unseen"] + n_iter["seen"]) / 25))
    reps, omega = class_stats(d, B)
    rob = {}
    for name, (Xtr, ytr) in {"plain": (X, d.target.to_numpy()), "crop": (X_ext, d_ext.target.to_numpy())}.items():
        t0 = time.perf_counter()
        bins = fit_bin_edges(Xtr)
        clf = HistGradientBoostingClassifier(max_iter=it, max_features=0.5, early_stopping=False, random_state=0, **PARAMS)
        clf.fit(apply_bin_edges(Xtr, bins), to_class(ytr, B))

        def pred_from(r, k):
            rows = Fte_rows[r[k:]]
            H = restream(rows)
            if k == 0:
                assert np.abs(H - X817[rows]).max() == 0.0
            return decide(clf.predict_proba(apply_bin_edges(H, bins)), reps, omega, True, B)

        full = [pred_from(r, 0) for r in te_recs]
        rng2 = np.random.default_rng(0)
        onset = [float(d_te.stop[r[-1]] + d_te.target[r[-1]]) for r in te_recs]
        out = {"k0": score_rows(d_te, np.concatenate(full))[0], "n_iter": it}
        for kname, k in KS.items():
            kk = [k if k is not None else int(rng2.integers(0, max(int((o - 60) // 5), 0) + 1)) for o in onset]
            kk = [min(q, len(r) - 1) for q, r in zip(kk, te_recs)]
            rows = np.concatenate([r[q:] for r, q in zip(te_recs, kk)])
            dr = d_te.iloc[rows].reset_index(drop=True)
            st, ct = score_rows(dr, np.concatenate([pred_from(r, q) for r, q in zip(te_recs, kk)]))
            sf, cf = score_rows(dr, np.concatenate([p[q:] for p, q in zip(full, kk)]))
            out[kname] = dict(truncated=st, full=sf, delta=st - sf, d_contrib=(np.array(ct) - np.array(cf)).tolist())
        rob[name] = out
        print(f"  HGB-only {name:5s} (n_iter {it}, {time.perf_counter() - t0:.0f} s): k0 {out['k0']:.2f} | "
              + " ".join(f"{kn} Δ {out[kn]['delta']:+.2f}" for kn in KS), flush=True)
    (OUT / "hgb_crop_precheck.json").write_text(json.dumps(dict(cv=cv, robustness=rob, time_s=time.perf_counter() - t_all), indent=2))
    print(f"saved {OUT / 'hgb_crop_precheck.json'} ({time.perf_counter() - t_all:.0f} s)")
