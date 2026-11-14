"""Work 2 step e: HGB B member with the 817 features + the 5 fold-specific CNN N2-detector summaries (n2det_feats.py),
dual CV, same classifier / class edges / decision / LOFO tree count as b2_ablation.py. Reference = B on the 817
features (ens_dyn_maxmin_spec_eye_B, 43.20).

Usage: python scripts/n2det_ablation.py [BASE]   (BASE "" = 817, "bip" = 817 + bipolar group)
  -> OUT/b2_<name>_{unseen,seen}_preds.npz, OUT/b2_<name>_B_proba.npz, merged into OUT/b2_ablation.json
     (name = n2det or bip+n2det)
"""
import json
import sys
import time

import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from b2_ablation import REF, b2_table
from dist_hgb import PARAMS, VARIANTS, run_clf
from dual_cv import OUT, seen_splits, unseen_splits
from n2det_feats import NAMES, fold_features
from range_breakdown import breakdown, load_preds

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else ""
    gname = f"{base}+n2det" if base else "n2det"
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X0, _ = build_table_ext(wg, sg)
    m = (df.split != "test").to_numpy()
    if base:
        X0 = np.concatenate([X0, b2_table(tuple(base.split("+")), df)[0]], axis=1)
    X0 = X0[m]
    d = df[m].reset_index(drop=True)
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    ref = breakdown(d, protocols, load_preds(REF, protocols))
    t0 = time.perf_counter()
    preds, n_iter, probas = {}, {}, {}
    for proto, splits in protocols.items():
        Xf = lambda k, proto=proto: np.concatenate([X0, fold_features(proto, k, recs)], axis=1)  # noqa: E731
        r = run_clf(Xf, d, splits, VARIANTS["B"], PARAMS, 400, keep_proba=True)["dist_continuous"]
        preds[proto], n_iter[proto], probas[proto] = r[0], r[1], r[3]
        np.savez(OUT / f"b2_{gname}_{proto}_preds.npz", *preds[proto])
        print(f"  {proto} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    np.savez(OUT / f"b2_{gname}_B_proba.npz", probas=np.array(probas, dtype=object), n_iter=np.array(n_iter, dtype=object))
    b = breakdown(d, protocols, preds)
    du = np.array(b["unseen"]["folds"]) - np.array(ref["unseen"]["folds"])
    ds = np.array(b["seen"]["folds"]) - np.array(ref["seen"]["folds"])
    nb = int((du < 0).sum() + (ds < 0).sum())
    delta = float(b["macro"]["wbmae"] - ref["macro"]["wbmae"])
    dc = (np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])).tolist()
    path = OUT / "b2_ablation.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[gname] = dict(n_new=len(NAMES), n_features=int(X0.shape[1] + len(NAMES)), unseen=float(b["unseen"]["wbmae"]),
                          seen=float(b["seen"]["wbmae"]), macro=float(b["macro"]["wbmae"]), delta=delta, better=nb,
                          d_unseen=du.tolist(), d_seen=ds.tolist(), contrib=np.array(b["macro"]["contrib"]).tolist(),
                          d_contrib=dc, n_iter=n_iter, time_s=time.perf_counter() - t0)
    path.write_text(json.dumps(summary, indent=2))
    print(f"{gname}: unseen {b['unseen']['wbmae']:.2f} seen {b['seen']['wbmae']:.2f} dual {float(b['macro']['wbmae']):.2f} "
          f"Δ vs B(817) {delta:+.2f} better {nb}/10 | Δ contrib " + " ".join(f"{x:+.2f}" for x in dc) +
          f" | n_iter {n_iter} ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"    folds Δ unseen {np.round(du, 2).tolist()} seen {np.round(ds, 2).tolist()}", flush=True)
