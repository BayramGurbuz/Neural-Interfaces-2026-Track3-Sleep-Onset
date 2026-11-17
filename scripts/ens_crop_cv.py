"""Candidate 2: HGB ensemble (c) (B, B_h1, B_h2, B2; ensemble_dist.MEMBERS) with N2-detector features AND stream-start
crop augmentation of the training table, dual CV at k = 0 (held-out rows = the original streams).

Training rows of every fold = the fold's original training rows + one start-cropped copy of each of its training
recordings (hgb_crop_precheck.cropped_copies, seed 0: 70 % k in 1..12 windows, 30 % k in 13..len-12): the 817 causal
features AND the detector summaries are re-streamed from window k (elapsed, EMAs, running extrema, first-2-min
differences, N2Summary all restart there); targets unchanged. Detector outputs are fold-specific and out-of-fold for the
training rows, as before (per-window values do not depend on the stream start, only their summaries do).
DET: old = shipped binary CNN detector (n2det_cnn_<proto>_f<k>.npy, 5 summaries of P);
     c   = multi-class detector, 3-seed mean, 5 summaries of closeness; cp = closeness + P (10 summaries).
     ctx6 = Work C context detector (last 6 windows; n2det_ctx_inner.py files), 3-seed mean, 5 summaries of closeness.
            Approximation: a cropped copy's first 5 windows keep the closeness computed with the pre-crop context (the
            detector models of the inner folds are not saved); the stream-from-window-2 check moved Spearman by 0.002.
Cache as ensemble_dist.member_probas: OUT/ens_<feat_set>_<member>.npz, feat_set = dyn_maxmin_spec_eye_n2<DET>_crop
(per-protocol .part.npz checkpoints), so the mix scripts can read the out-of-fold probabilities.

Usage: python scripts/ens_crop_cv.py old|c|cp [--seeds=0,1,2]   -> OUT/ens_crop_<DET>.json, OUT/ens_<feat_set>_(c)_{proto}_preds.npz
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import hgb_crop_precheck as pc
from dist_hgb import run_clf
from dual_cv import OUT, seen_splits, unseen_splits
from ensemble_dist import MEMBERS, ensemble_preds
from n2det_feats import summaries
from range_breakdown import breakdown, load_preds

KEYS = {"old": None, "c": ("close",), "cp": ("close", "p"), "ctx6": ("close",)}
SRC = {"c": "n2det_multi", "cp": "n2det_multi", "ctx6": "n2det_ctx6"}
# Tree caps: ensemble_dist.MEMBERS except B_h1 600 -> 400 (its LOFO choices with n2det were <= 225; the crop copies lengthened
# the B member by <= 1.6x in the pre-check). A fold that selects the cap is reported in the log (n_iter == cap).
CAPS = {"B_h1": 400}
REFS = {"HGB(c) 817": "ens_dyn_maxmin_spec_eye_(c)_B-B_h1-B_h2-B2@grid", "HGB(c)+n2det": "ens_dyn_maxmin_spec_eye_n2det_(c)",
        "HGB(c) n2c crop (cand 2)": "ens_dyn_maxmin_spec_eye_n2c_crop_(c)"}


def det_values(det, proto, k, seeds):
    """-> list of per-window arrays (one per summarised detector output) for outer fold k."""
    if det == "old":
        return [np.load(OUT / f"n2det_cnn_{proto}_f{k}.npy")]
    zs = [np.load(OUT / f"{SRC[det]}_s{s}_{proto}_f{k}.npz") for s in seeds]
    return [np.mean([z[key] for z in zs], axis=0) for key in KEYS[det]]


def run(det, seeds=(0, 1, 2), members=None, max_iter=None, tag=""):
    feat_set = f"dyn_maxmin_spec_eye_n2{det}_crop{tag}"
    m = (pc.df.split != "test").to_numpy()
    d = pc.df[m].reset_index(drop=True)
    X0 = pc.X817[m]
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    assert np.abs(pc.restream(recs[0]) - X0[recs[0]]).max() == 0.0
    c_rows, c_X, ks = pc.cropped_copies(recs, np.random.default_rng(0))
    crop_recs = [r[k:] for r, k in zip(recs, ks)]
    assert (np.concatenate(crop_recs) == c_rows).all()
    d_ext = pd.concat([d, d.iloc[c_rows]], ignore_index=True)
    tl_c = d.timeline.to_numpy()[c_rows]
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    print(f"{feat_set}: {len(d)} rows + {len(c_rows)} cropped-copy rows, {X0.shape[1]} + {5 * (1 if det == 'old' else len(KEYS[det]))} features", flush=True)

    def X_fold(proto, k):
        vals = det_values(det, proto, k, seeds)
        S = np.concatenate([summaries(v, recs) for v in vals], axis=1)
        Sc = np.concatenate([summaries(v, crop_recs)[c_rows] for v in vals], axis=1)
        return np.concatenate([np.concatenate([X0, S], axis=1), np.concatenate([c_X, Sc], axis=1)])

    def ext_splits(splits):
        out = []
        for tr, te in splits:
            tr_recs = np.unique(d.timeline.to_numpy()[tr])
            out.append((np.concatenate([tr, np.isin(tl_c, tr_recs)]), np.concatenate([te, np.zeros(len(c_rows), bool)])))
        return out

    t0 = time.perf_counter()
    probas, n_iters = {}, {}
    for name in (members or list(MEMBERS)):
        edges, params, mi = MEMBERS[name]
        mi = max_iter or CAPS.get(name, mi)
        path = OUT / f"ens_{feat_set}_{name}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            probas[name], n_iters[name] = z["probas"].item(), z["n_iter"].item()
            print(f"  member {name}: cached", flush=True)
            continue
        probas[name], n_iters[name] = {}, {}
        for proto, splits in protocols.items():
            part = OUT / f"ens_{feat_set}_{name}_{proto}.part.npz"
            if part.exists():
                z = np.load(part, allow_pickle=True)
                probas[name][proto] = [z[f"arr_{i}"] for i in range(len(splits))]
                n_iters[name][proto] = z["n_iter"].tolist()
                print(f"  member {name} {proto}: loaded from {part.name}", flush=True)
                continue
            t1 = time.perf_counter()
            Xf = lambda k, proto=proto: X_fold(proto, k)  # noqa: E731
            r = run_clf(Xf, d_ext, ext_splits(splits), edges, params, mi, keep_proba=True)["dist_continuous"]
            probas[name][proto], n_iters[name][proto] = r[3], r[1]
            np.savez(part, *r[3], n_iter=np.array(r[1]))
            print(f"  member {name} {proto}: n_iter {r[1]} (cap {mi}) ({time.perf_counter() - t1:.0f} s)", flush=True)
        np.savez(path, probas=np.array(probas[name], dtype=object), n_iter=np.array(n_iters[name], dtype=object))
    names = list(probas)
    preds = ensemble_preds(d, protocols, names, probas, "grid")
    for p in protocols:
        np.savez(OUT / f"ens_{feat_set}_(c)_{p}_preds.npz", *preds[p])
    b = breakdown(d, protocols, preds)
    res = dict(feat_set=feat_set, members=names, unseen=float(b["unseen"]["wbmae"]), seen=float(b["seen"]["wbmae"]),
               macro=float(b["macro"]["wbmae"]), contrib=np.array(b["macro"]["contrib"]).tolist(), n_iter=n_iters,
               time_s=time.perf_counter() - t0)
    msg = f"HGB({'+'.join(names)}) n2{det} + crop: unseen {res['unseen']:.2f} seen {res['seen']:.2f} dual {res['macro']:.2f}"
    for rn, rf in REFS.items():
        ref = breakdown(d, protocols, load_preds(rf, protocols))
        du = np.array(b["unseen"]["folds"]) - np.array(ref["unseen"]["folds"])
        ds = np.array(b["seen"]["folds"]) - np.array(ref["seen"]["folds"])
        dc = (np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])).tolist()
        res[f"vs_{rn}"] = dict(delta=float(b["macro"]["wbmae"] - ref["macro"]["wbmae"]), better=int((du < 0).sum() + (ds < 0).sum()),
                               d_unseen=du.tolist(), d_seen=ds.tolist(), d_contrib=dc)
        msg += (f"\n    vs {rn} ({float(ref['macro']['wbmae']):.2f}): Δ {res[f'vs_{rn}']['delta']:+.2f} better {res[f'vs_{rn}']['better']}/10 | Δ contrib "
                + " ".join(f"{x:+.2f}" for x in dc) + f" | folds Δ unseen {np.round(du, 2).tolist()} seen {np.round(ds, 2).tolist()}")
    (OUT / f"ens_crop_{det}{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    print(msg + f" ({res['time_s']:.0f} s)", flush=True)
    return res


if __name__ == "__main__":
    seeds = tuple(int(v) for v in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), "0,1,2").split(","))
    run(sys.argv[1], seeds)
