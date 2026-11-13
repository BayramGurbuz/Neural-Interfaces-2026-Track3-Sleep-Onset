"""B2: HGB ensemble (c) (B, B_h1, B_h2, B2 on the common grid) with the 817 features + a B2 group, dual CV.

Members are run exactly as ensemble_dist.member_probas (same params, max_iter, LOFO tree count) and cached as
OUT/ens_<feat_set>_<member>.npz with feat_set = dyn_maxmin_spec_eye_<group>. The B member was already run with
identical settings by b2_ablation.py (OUT/b2_<group>_B_proba.npz) and is reused.

Usage: python scripts/b2_ens.py GROUP   (e.g. bip)   -> OUT/ens_dyn_maxmin_spec_eye_<GROUP>_*.npz, b2_ens_<GROUP>.json
"""
import json
import sys
import time

import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from b2_ablation import b2_table
from dual_cv import OUT, seen_splits, unseen_splits
from ensemble_dist import MEMBERS, ensemble_preds, member_probas
from range_breakdown import breakdown, load_preds

if __name__ == "__main__":
    group = sys.argv[1]
    feat_set = f"dyn_maxmin_spec_eye_{group}"
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X0, _ = build_table_ext(wg, sg)
    Xg, gn = b2_table(tuple(group.split("+")), df)
    m = (df.split != "test").to_numpy()
    X = np.concatenate([X0, Xg], axis=1)[m]
    d = df[m].reset_index(drop=True)
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    print(f"{feat_set}: {X.shape[1]} features", flush=True)
    b_cache = OUT / f"ens_{feat_set}_B.npz"
    if not b_cache.exists():
        z = np.load(OUT / f"b2_{group}_B_proba.npz", allow_pickle=True)
        np.savez(b_cache, probas=z["probas"], n_iter=z["n_iter"])
        print(f"B member reused from b2_{group}_B_proba.npz", flush=True)
    t0 = time.perf_counter()
    probas, n_iters = {}, {}
    for name in MEMBERS:
        t1 = time.perf_counter()
        probas[name], n_iters[name] = member_probas(name, feat_set, X, d, protocols)
        print(f"  member {name}: n_iter {n_iters[name]} ({time.perf_counter() - t1:.0f} s)", flush=True)
    preds = ensemble_preds(d, protocols, list(MEMBERS), probas, "grid")
    for p in protocols:
        np.savez(OUT / f"ens_{feat_set}_(c)_{p}_preds.npz", *preds[p])
    ref = breakdown(d, protocols, load_preds("ens_dyn_maxmin_spec_eye_(c)_B-B_h1-B_h2-B2@grid", protocols))
    b = breakdown(d, protocols, preds)
    du = np.array(b["unseen"]["folds"]) - np.array(ref["unseen"]["folds"])
    ds = np.array(b["seen"]["folds"]) - np.array(ref["seen"]["folds"])
    res = dict(feat_set=feat_set, n_features=int(X.shape[1]), unseen=float(b["unseen"]["wbmae"]), seen=float(b["seen"]["wbmae"]),
               macro=float(b["macro"]["wbmae"]), delta=float(b["macro"]["wbmae"] - ref["macro"]["wbmae"]),
               better=int((du < 0).sum() + (ds < 0).sum()), d_unseen=du.tolist(), d_seen=ds.tolist(),
               d_contrib=(np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])).tolist(),
               n_iter={k: v for k, v in n_iters.items()}, time_s=time.perf_counter() - t0)
    (OUT / f"b2_ens_{group}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"HGB(c)+{group}: unseen {res['unseen']:.2f} seen {res['seen']:.2f} dual {res['macro']:.2f} "
          f"Δ vs HGB(c) 817 {res['delta']:+.2f} better {res['better']}/10 | Δ contrib "
          + " ".join(f"{x:+.2f}" for x in res["d_contrib"]) + f" ({res['time_s']:.0f} s)", flush=True)
