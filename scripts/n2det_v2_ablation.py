"""Work 3, stage 2 (CPU): HGB B member with the multi-class N2 detector's fold-specific outputs (n2det_v2_inner.py,
mean over the seeds given), dual CV, same classifier / class edges / decision / LOFO tree count as n2det_ablation.py.
Rows: "cp" = 817 + N2Summary of closeness (5) + N2Summary of P (5); "c" = 817 + N2Summary of closeness (5).
References: (1) B + the shipped binary detector's 5 summaries (b2_n2det, 40.86), and B on the 817 features (43.20).
Rule (user, 15 Nov): a row wins if Δ vs (1) <= -0.15 and it is better than (1) in >= 7/10 folds.

Usage: python scripts/n2det_v2_ablation.py [cp c] [--seeds=0,1,2]
  -> OUT/b2_n2multi_<row>_{unseen,seen}_preds.npz, OUT/b2_n2multi_<row>_B_proba.npz, OUT/n2det_v2_ablation.json
"""
import json
import sys
import time

import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from b2_ablation import REF
from dist_hgb import PARAMS, VARIANTS, run_clf
from dual_cv import OUT, seen_splits, unseen_splits
from n2det_feats import summaries
from range_breakdown import breakdown, load_preds

ROWS = {"cp": ("close", "p"), "c": ("close",)}

if __name__ == "__main__":
    rows = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ROWS)
    seeds = [int(v) for v in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), "0,1,2").split(",")]
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X0, _ = build_table_ext(wg, sg)
    m = (df.split != "test").to_numpy()
    X0 = X0[m]
    d = df[m].reset_index(drop=True)
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    refs = {"B(817)": breakdown(d, protocols, load_preds(REF, protocols)),
            "B+n2det": breakdown(d, protocols, load_preds("b2_n2det", protocols))}
    print(" | ".join(f"{k}: unseen {r['unseen']['wbmae']:.2f} seen {r['seen']['wbmae']:.2f} dual {float(r['macro']['wbmae']):.2f}"
                     for k, r in refs.items()), flush=True)
    path = OUT / "n2det_v2_ablation.json"
    summary = json.loads(path.read_text()) if path.exists() else {}

    def fold_features(proto, k, keys):
        zs = [np.load(OUT / f"n2det_multi_s{s}_{proto}_f{k}.npz") for s in seeds]
        return np.concatenate([summaries(np.mean([z[key] for z in zs], axis=0), recs) for key in keys], axis=1)

    for row in rows:
        gname = f"n2multi_{row}"
        t0 = time.perf_counter()
        preds, n_iter, probas = {}, {}, {}
        for proto, splits in protocols.items():
            Xf = lambda k, proto=proto: np.concatenate([X0, fold_features(proto, k, ROWS[row])], axis=1)  # noqa: E731
            r = run_clf(Xf, d, splits, VARIANTS["B"], PARAMS, 400, keep_proba=True)["dist_continuous"]
            preds[proto], n_iter[proto], probas[proto] = r[0], r[1], r[3]
            np.savez(OUT / f"b2_{gname}_{proto}_preds.npz", *preds[proto])
            print(f"  {gname} {proto} done ({time.perf_counter() - t0:.0f} s)", flush=True)
        np.savez(OUT / f"b2_{gname}_B_proba.npz", probas=np.array(probas, dtype=object), n_iter=np.array(n_iter, dtype=object))
        b = breakdown(d, protocols, preds)
        entry = dict(seeds=seeds, n_features=int(X0.shape[1] + 5 * len(ROWS[row])), unseen=float(b["unseen"]["wbmae"]),
                     seen=float(b["seen"]["wbmae"]), macro=float(b["macro"]["wbmae"]), contrib=np.array(b["macro"]["contrib"]).tolist(),
                     n_iter=n_iter, time_s=time.perf_counter() - t0)
        msg = f"{gname}: unseen {entry['unseen']:.2f} seen {entry['seen']:.2f} dual {entry['macro']:.2f}"
        for rn, ref in refs.items():
            du = np.array(b["unseen"]["folds"]) - np.array(ref["unseen"]["folds"])
            ds = np.array(b["seen"]["folds"]) - np.array(ref["seen"]["folds"])
            dc = (np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])).tolist()
            entry[f"vs_{rn}"] = dict(delta=float(b["macro"]["wbmae"] - ref["macro"]["wbmae"]), better=int((du < 0).sum() + (ds < 0).sum()),
                                     d_unseen=du.tolist(), d_seen=ds.tolist(), d_contrib=dc)
            e = entry[f"vs_{rn}"]
            msg += (f"\n    vs {rn}: Δ {e['delta']:+.2f} better {e['better']}/10 | Δ contrib " + " ".join(f"{x:+.2f}" for x in dc)
                    + f" | folds Δ unseen {np.round(du, 2).tolist()} seen {np.round(ds, 2).tolist()}")
        e = entry["vs_B+n2det"]
        entry["wins"] = bool(e["delta"] <= -0.15 and e["better"] >= 7)
        summary[gname] = entry
        path.write_text(json.dumps(summary, indent=2))
        print(msg + f"\n    n_iter {n_iter}; wins (Δ <= -0.15 and >= 7/10 vs B+n2det): {entry['wins']} ({entry['time_s']:.0f} s)", flush=True)
