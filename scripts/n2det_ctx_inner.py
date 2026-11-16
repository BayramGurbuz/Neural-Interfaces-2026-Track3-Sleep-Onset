"""Work C, stage 2 (GPU): fold-specific outputs of the context N2 detector (n2det_ctx.py, last N windows) for the downstream
HGB ablation, one seed per call. Same leakage rule as n2det_v2_inner.py: in every outer fold (5 unseen + 5 seen) the detector
sees only that fold's training recordings; held-out rows get the outer detector's outputs (LOFO epoch on the other folds'
held-out binary CE), training rows get out-of-fold outputs from 5 inner folds at that epoch (grouped by subject in the unseen
protocol, by recording in the seen protocol). Context from the recording start (k = 0); a cropped copy's first N-1 windows
would see a shorter context (not modelled in the ablation, which uses no crop copies).
Outputs per window: p = P(post-onset class), close = expected class index / 4.

Usage: python scripts/n2det_ctx_inner.py N SEED  -> OUT/n2det_ctx<N>_s<SEED>_<proto>_f<k>.npz (p, close; resumable),
       OUT/n2det_ctx<N>_s<SEED>.json (epochs)
"""
import json
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from n2det_ctx import Data, load_all, run_fold
from dual_cv import OUT, seen_splits, unseen_splits


def fill(arr_p, arr_c, rows, o):
    arr_p[rows], arr_c[rows] = o["p"], o["close"]


if __name__ == "__main__":
    n_ctx, seed = int(sys.argv[1]), int(sys.argv[2])
    d, _, post, m, mp = load_all()
    data = Data(d, post, m, mp)
    tl = d.timeline.to_numpy()
    tls = lambda mask: list(dict.fromkeys(tl[mask]))  # noqa: E731
    info = {}
    t_all = time.perf_counter()
    for proto, splits in {"unseen": unseen_splits(d), "seen": seen_splits(d)}.items():
        files = [OUT / f"n2det_ctx{n_ctx}_s{seed}_{proto}_f{k}.npz" for k in range(len(splits))]
        if all(f.exists() for f in files):
            print(f"{proto}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        runs = [run_fold(data, tls(tr), tls(te), n_ctx, seed) for tr, te in splits]
        C = np.array([[e["ce"] for e in r[0]] for r in runs])
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
        info[proto] = dict(epochs=[s + 1 for s in sel], ce_out_sel=[float(C[k, s]) for k, s in enumerate(sel)])
        print(f"{proto} outer: epochs {info[proto]['epochs']} ({time.perf_counter() - t0:.0f} s)", flush=True)
        for k, (tr, te) in enumerate(splits):
            if files[k].exists():
                continue
            t0 = time.perf_counter()
            p, close = np.full(len(d), np.nan, np.float32), np.full(len(d), np.nan, np.float32)
            out, rows = runs[k]
            fill(p, close, rows, out[sel[k]])
            rows_tr = np.flatnonzero(tr)
            groups = (d.subject.to_numpy() if proto == "unseen" else tl)[rows_tr]
            for itr, ite in GroupKFold(5).split(rows_tr, groups=groups):
                mtr, mte = np.zeros(len(d), bool), np.zeros(len(d), bool)
                mtr[rows_tr[itr]] = True
                mte[rows_tr[ite]] = True
                o, r = run_fold(data, tls(mtr), tls(mte), n_ctx, seed, stop_epoch=sel[k] + 1)
                fill(p, close, r, o[-1])
            assert not np.isnan(p).any() and not np.isnan(close).any()
            np.savez(files[k], p=p, close=close)
            print(f"  inner {proto} fold {k}: {time.perf_counter() - t0:.0f} s", flush=True)
    path = OUT / f"n2det_ctx{n_ctx}_s{seed}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(info)
    path.write_text(json.dumps(old, indent=2))
    print(f"ctx{n_ctx} seed {seed} done ({time.perf_counter() - t_all:.0f} s)", flush=True)
