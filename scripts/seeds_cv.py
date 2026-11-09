"""Seed-count study for the GRU (config uw): 10 seeds per fold in one dual-CV run; the averages of
the first 3 / 5 / 10 seeds are scored like gru_cv.run_protocol (epoch per fold chosen LOFO) and
saved as gru_uw_n{3,5,10}_{protocol}_{preds,proba}.npz, so mix_gru_hgb.py can mix them with HGB(c).
Training stops at epoch 40 (same cosine schedule as uw; selection was always 10-15).

Usage: python scripts/seeds_cv.py
"""
import json
import time

import numpy as np

from dist_hgb import class_stats, decide
from dual_cv import OUT, dual_report, fmt, seen_splits, unseen_splits
from gru_cv import CONFIGS, EDGES, load_inputs, run_fold
from stream_utils import summarize

NS = (3, 5, 10)

if __name__ == "__main__":
    d, recs, F, gain = load_inputs()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    cfg = CONFIGS["uw_n10"]
    ee = cfg["eval_every"]
    res_all = {n: {"preds": {}, "probas": {}, "ep": {}} for n in NS}
    t_all = time.perf_counter()
    for proto, splits in protocols.items():
        curves = {n: [] for n in NS}
        cands = {n: [] for n in NS}
        probs = {n: [] for n in NS}
        for k, (tr, te) in enumerate(splits):
            t0 = time.perf_counter()
            runs = [run_fold(d, recs, F, gain, tr, te, cfg, s) for s in range(max(NS))]
            dte = d[te].reset_index(drop=True)
            _, omega = class_stats(d[tr].reset_index(drop=True), EDGES)
            for n in NS:
                Pk = [np.mean([r[j] for r in runs[:n]], axis=0) for j in range(len(runs[0]))]
                q = [decide(P.astype(np.float64), None, omega, True, EDGES).astype(np.float32) for P in Pk]
                curves[n].append([summarize(dte, x)["wbmae"] for x in q])
                cands[n].append(q)
                probs[n].append(Pk)
            print(f"  {proto} fold {k}: {time.perf_counter() - t0:.0f} s", flush=True)
        for n in NS:
            C = np.array(curves[n])
            sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
            res_all[n]["preds"][proto] = [cands[n][k][j] for k, j in enumerate(sel)]
            res_all[n]["probas"][proto] = [probs[n][k][j] for k, j in enumerate(sel)]
            res_all[n]["ep"][proto] = [ee * (j + 1) for j in sel]
            np.savez(OUT / f"gru_uw_n{n}_{proto}_preds.npz", *res_all[n]["preds"][proto])
            np.savez(OUT / f"gru_uw_n{n}_{proto}_proba.npz", *res_all[n]["probas"][proto])
    summary_path = OUT / "gru.json"
    summary = json.loads(summary_path.read_text())
    for n in NS:
        r = res_all[n]
        res = dual_report(f"GRU-uw_n{n}", d, protocols, r["preds"], dict(n_seeds=n, epochs=r["ep"]))
        summary[f"GRU-uw_n{n}"] = res
        print(fmt(res), f" epochs {r['ep']}", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"total {time.perf_counter() - t_all:.0f} s")
