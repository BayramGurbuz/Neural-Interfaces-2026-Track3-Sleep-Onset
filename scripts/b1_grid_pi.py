"""B1a: onset-dependent grid weight pi(D) in the 30 s grid decision (no retraining).

In the public data every onset D <= 900 s is a multiple of 30 s after the recording start; above 900 s
only ~57 % are. Instead of one constant pi, the mass of every 1 s cell of remaining time y is moved to
the grid with weight pi(D), D = t + y (t = window stop), pi(D) = fraction of on-grid onsets among the
fold's TRAINING recordings in the D range of that cell.

Variants (reference = constant pi 0.86, the submitted mix3_540_grid recipe, dual CV 38.35):
  const086 : pi = 0.86 everywhere (reference, recomputed here)
  step     : ranges (0, 900], (900, inf)
  bins4    : ranges (0, 900], (900, 1100], (1100, 1300], (1300, inf)
  lofo     : nested-LOFO choice among const086 / step / bins4
Same weights, alpha, structure and scoring as decision_layer.py (official binned_mae, nested LOFO).

Usage: python scripts/b1_grid_pi.py   (writes OUT/b1_grid_pi.json, OUT/b1pi_<variant>_*_preds.npz)
"""
import json

import numpy as np

from decision_layer import (ATOM, NF, NS, T_MAX, build_folds, evaluate, official_rec_scores, predict,
                            rec_onsets, select, slot_median, test_fold, by_proto)
from dual_cv import OUT, load_trainval, seen_splits, unseen_splits
from range_breakdown import load_preds

VARIANTS = {"step": [900.0], "bins4": [900.0, 1100.0, 1300.0]}
REF_NAME = "declayer_E2_grid_pi_0_86"


def on_grid(D):
    return np.isclose(D % 30, 0) | np.isclose(D % 30, 30)


def pi_table(D_train, cuts):
    """(T_MAX, NF): pi of the 1 s cell j for a window ending at 5 * t_idx s (D = t + j + 0.5)."""
    edges = np.concatenate([[-np.inf], cuts, [np.inf]])
    g = on_grid(D_train)
    ratio = np.array([g[(D_train > lo) & (D_train <= hi)].mean() if ((D_train > lo) & (D_train <= hi)).any() else g.mean()
                      for lo, hi in zip(edges[:-1], edges[1:])])
    D = 5.0 * np.arange(T_MAX)[:, None] + np.arange(NF)[None, :] + 0.5
    return ratio[np.searchsorted(edges[1:-1], D, side="left")], ratio


def to_slots_pi(fine, t, pi_cells):
    """As decision_layer.to_slots, with a per-cell grid weight pi_cells (N, NF)."""
    S = np.zeros((len(fine), NS))
    S[:, 1::2] = (1.0 - pi_cells) * fine[:, :NF]
    r = np.round(t).astype(int) % 30
    moved = pi_cells * fine[:, :NF]
    for rr in np.unique(r):
        m = r == rr
        S[m] += moved[m] @ ATOM[int(rr)]
    S[:, NS - 1] += fine[:, NF]
    return S


def predict_pi(f, tab, alpha=0.7, chunk=4096):
    """Reference decision (old class weights, component structure) with pi(D) from ``tab``."""
    out = np.empty(len(f["t"]))
    ws, wh = f["w_old"]
    for i0 in range(0, len(out), chunk):
        sl = slice(i0, i0 + chunk)
        t = f["t"][sl]
        pc = tab[np.round(t / 5.0).astype(int)]
        seq = [to_slots_pi(x[sl], t, pc) for x in f["seq"]]
        hgb = to_slots_pi(f["hgb"][sl], t, pc)
        q = np.mean([(s * ws) / (s * ws).sum(1, keepdims=True) for s in seq], axis=0)
        h = hgb * wh
        out[sl] = slot_median(alpha * q + (1 - alpha) * h / h.sum(1, keepdims=True))
    return out


def compare(e, r0):
    du = np.array(e["unseen"]["folds"]) - np.array(r0["unseen"]["folds"])
    ds = np.array(e["seen"]["folds"]) - np.array(r0["seen"]["folds"])
    return du, ds, int((du < -1e-9).sum() + (ds < -1e-9).sum())


if __name__ == "__main__":
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    folds = build_folds(d, protocols)
    onset = rec_onsets(d)
    ratios = {}
    for f, (proto, splits) in zip(folds, [(p, s) for p, ss in protocols.items() for s in ss]):
        D_tr = onset.loc[d[splits[0]].timeline.unique()].to_numpy()
        f["tabs"] = {}
        for v, cuts in VARIANTS.items():
            f["tabs"][v], r = pi_table(D_tr, cuts)
            ratios.setdefault(v, []).append(r.round(3).tolist())
    for v, rs in ratios.items():
        print(f"in-fold pi per range, {v}: {rs}", flush=True)

    ref_saved = load_preds(REF_NAME, protocols)
    V = {"const086": [predict(f, "old", 0.86) for f in folds]}
    dev = max(float(np.abs(a - ref_saved[f["proto"]][f["k"]]).max()) for a, f in zip(V["const086"], folds))
    print(f"reference const pi 0.86 reproduced: max |Δ| vs saved = {dev:.2e} s", flush=True)
    for v in VARIANTS:
        V[v] = [predict_pi(f, f["tabs"][v]) for f in folds]
        print("  done", v, flush=True)
    V["lofo"], choice, allc = select(folds, {k: V[k] for k in ("const086", "step", "bins4")})
    print(f"  lofo choices {choice}, all-fold {allc}", flush=True)

    E = {k: evaluate(folds, v) for k, v in V.items()}
    r0 = E["const086"]
    res = {}
    print(f"\n{'variant':10s} unseen   seen  macro      Δ  better | Δ contrib [0,40) [40,90) [90,300) [300,600] |"
          f" on-grid (Δ)    off-grid (Δ)    onset<=330 (Δ)")
    for k, e in E.items():
        du, ds, nb = compare(e, r0)
        dc = np.array(e["macro"]["contrib"]) - np.array(r0["macro"]["contrib"])
        res[k] = dict(unseen=e["unseen"]["mean"], seen=e["seen"]["mean"], macro=e["macro"]["mean"],
                      delta=e["macro"]["mean"] - r0["macro"]["mean"], better=nb, d_unseen=du.tolist(), d_seen=ds.tolist(),
                      contrib=e["macro"]["contrib"], d_contrib=dc.tolist(),
                      on_grid={p: e[p]["on_grid"] for p in ("unseen", "seen", "macro")},
                      off_grid={p: e[p]["off_grid"] for p in ("unseen", "seen", "macro")},
                      short={p: e[p]["short"] for p in ("unseen", "seen", "macro")})
        print(f"{k:10s} {e['unseen']['mean']:6.2f} {e['seen']['mean']:6.2f} {e['macro']['mean']:6.2f} {res[k]['delta']:+6.2f}"
              f"  {nb:2d}/10 | " + " ".join(f"{x:+8.2f}" for x in dc) +
              f" | {e['macro']['on_grid']:6.2f} ({e['macro']['on_grid'] - r0['macro']['on_grid']:+.2f})"
              f"  {e['macro']['off_grid']:6.2f} ({e['macro']['off_grid'] - r0['macro']['off_grid']:+.2f})"
              f"  {e['macro']['short']:6.2f} ({e['macro']['short'] - r0['macro']['short']:+.2f})")
        print(f"{'':10s} folds Δ unseen {np.round(du, 2).tolist()} seen {np.round(ds, 2).tolist()}")
    # vs the pre-grid mix3 (39.40) for context
    base = load_preds("mix3_gru_tcn_cnnemb", protocols)
    e_b = evaluate(folds, [base[f["proto"]][f["k"]] for f in folds])
    print(f"\ncontext: mix3 without grid {e_b['macro']['mean']:.2f}; off-grid {e_b['macro']['off_grid']:.2f}, "
          f"on-grid {e_b['macro']['on_grid']:.2f}")
    for k, v in V.items():
        if k != "const086":
            P = by_proto(folds, v)
            for p in P:
                np.savez(OUT / f"b1pi_{k}_{p}_preds.npz", *P[p])

    print("\n== public test (40 recordings, train+val model, scored once)", flush=True)
    ft = test_fold()
    D_tv = rec_onsets(d).to_numpy()
    test = {"const086": float(official_rec_scores(ft["dte"], predict(ft, "old", 0.86)).mean())}
    for v, cuts in VARIANTS.items():
        tab, r = pi_table(D_tv, cuts)
        test[v] = float(official_rec_scores(ft["dte"], predict_pi(ft, tab)).mean())
        print(f"  pi ranges {v}: {r.round(3).tolist()}")
    for k, s in test.items():
        print(f"  {k:10s} {s:.4f} ({s - test['const086']:+.2f})")
    (OUT / "b1_grid_pi.json").write_text(json.dumps(dict(rows=res, test=test, ratios=ratios, lofo=dict(folds=choice, all=allc),
                                                         no_grid=dict(macro=e_b["macro"]["mean"], off=e_b["macro"]["off_grid"],
                                                                      on=e_b["macro"]["on_grid"])), indent=2))
