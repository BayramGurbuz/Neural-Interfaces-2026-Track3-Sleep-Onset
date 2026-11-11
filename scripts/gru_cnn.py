"""Stage 2 of the CNN experiment: GRU-uw on the 125 feature inputs + per-window CNN encoder outputs
(cnn_encoder.py), dual CV. Seed s of the GRU uses the fold's seed-s encoder; training recordings get
in-sample encoder outputs, held-out recordings out-of-sample ones (the encoder never saw them).

kind "emb": 32-d embedding (+ its first-2-min difference) -> 125 + 64 inputs.
kind "logits": 11 class logits (+ difference) -> 125 + 22 inputs (for the split-half variant).
Saved as gru_<out>_{proto}_{preds,proba}.npz (usable by mix_seq.py / range_breakdown.py).

Usage: python scripts/gru_cnn.py CNN_NAME [emb|logits] [n_seeds] [out_name]
"""
import json
import sys
import time

import numpy as np

from dist_hgb import class_stats, decide
from dual_cv import OUT, dual_report, fmt, seen_splits, unseen_splits
from gru_cv import CONFIGS, EDGES, load_inputs, run_fold
from range_breakdown import breakdown, load_preds, print_breakdown
from stream_utils import summarize

if __name__ == "__main__":
    cnn = sys.argv[1]
    kind = sys.argv[2] if len(sys.argv) > 2 else "emb"
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out = sys.argv[4] if len(sys.argv) > 4 else f"{cnn}_{kind}"
    cfg = dict(CONFIGS["uw"], run_epochs=40, n_seeds=n_seeds)
    d, recs, F, gain = load_inputs()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    t_all = time.perf_counter()
    preds, eps = {}, {}
    for proto, splits in protocols.items():
        curves, cands, probs = [], [], []
        for k, (tr, te) in enumerate(splits):
            t0 = time.perf_counter()
            runs = []
            for s in range(n_seeds):
                z = np.load(OUT / f"cnn_{cnn}_{proto}_f{k}_s{s}.npz")
                extra = z[kind].astype(np.float32)
                assert len(extra) == len(d)
                runs.append(run_fold(d, recs, F, gain, tr, te, cfg, s, extra=extra))
            dte = d[te].reset_index(drop=True)
            _, omega = class_stats(d[tr].reset_index(drop=True), EDGES)
            Pk = [np.mean([r[j] for r in runs], axis=0) for j in range(len(runs[0]))]
            q = [decide(P.astype(np.float64), None, omega, True, EDGES).astype(np.float32) for P in Pk]
            curves.append([summarize(dte, x)["wbmae"] for x in q])
            cands.append(q)
            probs.append(Pk)
            print(f"  {out} {proto} fold {k}: {time.perf_counter() - t0:.0f} s, curve " +
                  " ".join(f"{x:.1f}" for x in curves[-1]), flush=True)
        C = np.array(curves)
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
        preds[proto] = [cands[k][j] for k, j in enumerate(sel)]
        eps[proto] = [cfg["eval_every"] * (j + 1) for j in sel]
        np.savez(OUT / f"gru_{out}_{proto}_preds.npz", *preds[proto])
        np.savez(OUT / f"gru_{out}_{proto}_proba.npz", *[probs[k][j] for k, j in enumerate(sel)])
    res = dual_report(out, d, protocols, preds, dict(epochs=eps, cfg=cfg, cnn=cnn, kind=kind,
                                                     time_s=time.perf_counter() - t_all))
    summary_path = OUT / "gru.json"
    summary = json.loads(summary_path.read_text())
    summary[out] = res
    summary_path.write_text(json.dumps(summary, indent=2))
    print(fmt(res), f" epochs {eps}  ({res['time_s']:.0f} s)", flush=True)
    b = breakdown(d, protocols, preds)
    for ref in (f"gru_uw_n{n_seeds}", "gru_uw_n10"):
        print_breakdown(f"GRU+{out} vs {ref}", b, breakdown(d, protocols, load_preds(ref, protocols)))
