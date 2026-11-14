"""Work 2 step e: HGB ensemble (c) (B, B_h1, B_h2, B2, same settings as ensemble_dist.MEMBERS) with the 817 features
[+ BASE group] + the 5 fold-specific CNN N2-detector summaries; dual CV. The B member comes from n2det_ablation.py.
Cache format as ensemble_dist.member_probas (OUT/ens_<feat_set>_<member>.npz) so b2_mix.py can use it.

Usage: python scripts/n2det_ens.py [BASE]   -> feat_set dyn_maxmin_spec_eye_n2det (or _<BASE>+n2det)
"""
import json
import sys
import time

import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from b2_ablation import b2_table
from dist_hgb import run_clf
from dual_cv import OUT, seen_splits, unseen_splits
from ensemble_dist import MEMBERS, ensemble_preds
from n2det_feats import fold_features
from range_breakdown import breakdown, load_preds

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else ""
    gname = f"{base}+n2det" if base else "n2det"
    feat_set = f"dyn_maxmin_spec_eye_{gname}"
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X0, _ = build_table_ext(wg, sg)
    m = (df.split != "test").to_numpy()
    if base:
        X0 = np.concatenate([X0, b2_table(tuple(base.split("+")), df)[0]], axis=1)
    X0 = X0[m]
    d = df[m].reset_index(drop=True)
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    b_cache = OUT / f"ens_{feat_set}_B.npz"
    if not b_cache.exists():
        z = np.load(OUT / f"b2_{gname}_B_proba.npz", allow_pickle=True)
        np.savez(b_cache, probas=z["probas"], n_iter=z["n_iter"])
    t0 = time.perf_counter()
    probas, n_iters = {}, {}
    for name, (edges, params, max_iter) in MEMBERS.items():
        path = OUT / f"ens_{feat_set}_{name}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            probas[name], n_iters[name] = z["probas"].item(), z["n_iter"].item()
            continue
        probas[name], n_iters[name] = {}, {}
        for proto, splits in protocols.items():
            part = OUT / f"ens_{feat_set}_{name}_{proto}.part.npz"  # per-protocol checkpoint (survives a shutdown)
            if part.exists():
                z = np.load(part, allow_pickle=True)
                probas[name][proto] = [z[f"arr_{i}"] for i in range(len(splits))]
                n_iters[name][proto] = z["n_iter"].tolist()
                print(f"  member {name} {proto}: loaded from {part.name}", flush=True)
                continue
            t1 = time.perf_counter()
            Xf = lambda k, proto=proto: np.concatenate([X0, fold_features(proto, k, recs)], axis=1)  # noqa: E731
            r = run_clf(Xf, d, splits, edges, params, max_iter, keep_proba=True)["dist_continuous"]
            probas[name][proto], n_iters[name][proto] = r[3], r[1]
            np.savez(part, *r[3], n_iter=np.array(r[1]))
            print(f"  member {name} {proto}: n_iter {r[1]} ({time.perf_counter() - t1:.0f} s)", flush=True)
        np.savez(path, probas=np.array(probas[name], dtype=object), n_iter=np.array(n_iters[name], dtype=object))
    preds = ensemble_preds(d, protocols, list(MEMBERS), probas, "grid")
    for p in protocols:
        np.savez(OUT / f"ens_{feat_set}_(c)_{p}_preds.npz", *preds[p])
    ref = breakdown(d, protocols, load_preds("ens_dyn_maxmin_spec_eye_(c)_B-B_h1-B_h2-B2@grid", protocols))
    b = breakdown(d, protocols, preds)
    du = np.array(b["unseen"]["folds"]) - np.array(ref["unseen"]["folds"])
    ds = np.array(b["seen"]["folds"]) - np.array(ref["seen"]["folds"])
    res = dict(feat_set=feat_set, unseen=float(b["unseen"]["wbmae"]), seen=float(b["seen"]["wbmae"]),
               macro=float(b["macro"]["wbmae"]), delta=float(b["macro"]["wbmae"] - ref["macro"]["wbmae"]),
               better=int((du < 0).sum() + (ds < 0).sum()), d_unseen=du.tolist(), d_seen=ds.tolist(),
               d_contrib=(np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])).tolist(),
               n_iter=n_iters, time_s=time.perf_counter() - t0)
    (OUT / f"b2_ens_{gname}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"HGB(c)+{gname}: unseen {res['unseen']:.2f} seen {res['seen']:.2f} dual {res['macro']:.2f} "
          f"Δ vs HGB(c) 817 {res['delta']:+.2f} better {res['better']}/10 | Δ contrib "
          + " ".join(f"{x:+.2f}" for x in res["d_contrib"]) + f" ({res['time_s']:.0f} s)", flush=True)
