"""Experiment 3: other causal sequence models (LSTM, TCN) with the GRU-uw recipe, 10 seeds per fold.

Same protocol as seeds_cv.py: probabilities averaged over the seeds at every checkpoint (every
eval_every epochs), checkpoint of fold k chosen LOFO within the protocol; saved as
gru_<config>_{protocol}_{preds,proba}.npz so mix_gru_hgb.py / range_breakdown.py can use them.

Usage: python scripts/seq_cv.py lstm_n10 tcn_n10
"""
import json
import sys
import time

import numpy as np

from dist_hgb import class_stats, decide
from dual_cv import OUT, dual_report, fmt, seen_splits, unseen_splits
from gru_cv import CONFIGS, EDGES, load_inputs, run_fold
from stream_utils import summarize

if __name__ == "__main__":
    d, recs, F, gain = load_inputs()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    summary_path = OUT / "gru.json"
    for name in sys.argv[1:]:
        cfg = CONFIGS[name]
        t_all = time.perf_counter()
        preds, eps = {}, {}
        for proto, splits in protocols.items():
            curves, cands, probs = [], [], []
            for k, (tr, te) in enumerate(splits):
                t0 = time.perf_counter()
                runs = [run_fold(d, recs, F, gain, tr, te, cfg, s) for s in range(cfg["n_seeds"])]
                dte = d[te].reset_index(drop=True)
                _, omega = class_stats(d[tr].reset_index(drop=True), EDGES)
                Pk = [np.mean([r[j] for r in runs], axis=0) for j in range(len(runs[0]))]
                q = [decide(P.astype(np.float64), None, omega, True, EDGES).astype(np.float32) for P in Pk]
                curves.append([summarize(dte, x)["wbmae"] for x in q])
                cands.append(q)
                probs.append(Pk)
                print(f"  {name} {proto} fold {k}: {time.perf_counter() - t0:.0f} s, curve " +
                      " ".join(f"{x:.1f}" for x in curves[-1]), flush=True)
            C = np.array(curves)
            sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
            preds[proto] = [cands[k][j] for k, j in enumerate(sel)]
            eps[proto] = [cfg["eval_every"] * (j + 1) for j in sel]
            np.savez(OUT / f"gru_{name}_{proto}_preds.npz", *preds[proto])
            np.savez(OUT / f"gru_{name}_{proto}_proba.npz", *[probs[k][j] for k, j in enumerate(sel)])
        res = dual_report(name, d, protocols, preds, dict(epochs=eps, cfg=cfg, time_s=time.perf_counter() - t_all))
        summary = json.loads(summary_path.read_text())
        summary[name] = res
        summary_path.write_text(json.dumps(summary, indent=2))
        print(fmt(res), f" epochs {eps}  ({res['time_s']:.0f} s)", flush=True)
