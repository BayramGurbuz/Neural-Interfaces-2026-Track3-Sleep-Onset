"""B2 feature-group ablation with the HGB B member only (dual CV, same classifier / class edges / decision /
LOFO tree count as the B member of ensemble (c)). Reference = B on the 817 features (dyn_maxmin_spec_eye),
saved OOF predictions ens_dyn_maxmin_spec_eye_B_{proto}_preds.npz (dual CV 43.20).

Groups (features_b2.py): bip, events, epoch; any "+"-joined combination (e.g. bip+events).
New window-level groups enter as current + EMA 30 s / 2 min + first-2-min differences (B2Stream).
A group "wins" if Δ <= -0.15 and it is better in >= 7/10 folds.

Usage: python scripts/b2_ablation.py GROUP [GROUP ...]
  writes OUT/b2_<group>_{unseen,seen}_preds.npz, OUT/b2_<group>_B_proba.npz, OUT/b2_ablation.json (merged)
"""
import json
import sys
import time

import numpy as np

from ablation_c import ABLATIONS, build_table_ext
from causal_features import CORE, PER_CHANNEL, core_index
from dist_hgb import PARAMS, VARIANTS, run_clf
from dual_cv import OUT, seen_splits, unseen_splits
from features_b2 import B2Stream, b2_stream_names
from range_breakdown import breakdown, load_preds
from train_hgb import SAVE

REF = "ens_dyn_maxmin_spec_eye_B"


def b2_table(groups, df):
    """Stream the B2 groups over every recording (time order) -> (len(df), n) features + names."""
    zb, z2 = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_b2.npz")
    assert (z2["timeline"] == zb["timeline"]).all() and (z2["stop"] == zb["stop"]).all()
    assert (df.timeline.to_numpy() == zb["timeline"]).all() and (df.stop.to_numpy() == zb["stop"]).all()
    ch = [str(c) for c in zb["ch_names"]]
    win = [g for g in groups if g in ("bip", "events")]
    W = np.concatenate([z2[f"feats_{g}"] for g in win], axis=1).astype(np.float64) if win else np.zeros((len(df), 0))
    wnames = [str(n) for g in win for n in z2[f"names_{g}"]]
    core = zb["feats"][:, core_index(ch)].astype(np.float64)
    core_names = [f"{c}_{f}" for c in ch for f in CORE]
    st = B2Stream(5.0, epoch_group="epoch" in groups)
    names = b2_stream_names(wnames, core_names, "epoch" in groups)
    out = np.empty((len(df), len(names)), np.float32)
    order = np.lexsort((df.stop.to_numpy(), df.timeline.to_numpy()))
    prev = None
    tl = df.timeline.to_numpy()
    for i in order:
        if tl[i] != prev:
            st.reset()
            prev = tl[i]
        out[i] = st.update(W[i], core[i])
    return out, names


if __name__ == "__main__":
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X0, names0 = build_table_ext(wg, sg)
    m = (df.split != "test").to_numpy()
    d = df[m].reset_index(drop=True)
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    ref = breakdown(d, protocols, load_preds(REF, protocols))
    print(f"reference B (817): unseen {ref['unseen']['wbmae']:.2f} seen {ref['seen']['wbmae']:.2f} "
          f"dual {float(ref['macro']['wbmae']):.2f}", flush=True)
    path = OUT / "b2_ablation.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for gname in sys.argv[1:]:
        groups = tuple(gname.split("+"))
        t0 = time.perf_counter()
        Xg, gn = b2_table(groups, df)
        X = np.concatenate([X0, Xg], axis=1)[m]
        print(f"=== {gname}: +{len(gn)} features -> {X.shape[1]} (built in {time.perf_counter() - t0:.0f} s); "
              f"NaN {int(np.isnan(Xg).sum())}", flush=True)
        preds, n_iter, probas = {}, {}, {}
        for proto, splits in protocols.items():
            r = run_clf(X, d, splits, VARIANTS["B"], PARAMS, 400, keep_proba=True)["dist_continuous"]
            preds[proto], n_iter[proto], probas[proto] = r[0], r[1], r[3]
            np.savez(OUT / f"b2_{gname}_{proto}_preds.npz", *preds[proto])
        np.savez(OUT / f"b2_{gname}_B_proba.npz", probas=np.array(probas, dtype=object), n_iter=np.array(n_iter, dtype=object))
        b = breakdown(d, protocols, preds)
        du = np.array(b["unseen"]["folds"]) - np.array(ref["unseen"]["folds"])
        ds = np.array(b["seen"]["folds"]) - np.array(ref["seen"]["folds"])
        nb = int((du < 0).sum() + (ds < 0).sum())
        delta = float(b["macro"]["wbmae"] - ref["macro"]["wbmae"])
        dc = (np.array(b["macro"]["contrib"]) - np.array(ref["macro"]["contrib"])).tolist()
        wins = delta <= -0.15 and nb >= 7
        summary[gname] = dict(n_new=len(gn), n_features=int(X.shape[1]), unseen=float(b["unseen"]["wbmae"]),
                              seen=float(b["seen"]["wbmae"]), macro=float(b["macro"]["wbmae"]), delta=delta, better=nb,
                              d_unseen=du.tolist(), d_seen=ds.tolist(), contrib=np.array(b["macro"]["contrib"]).tolist(),
                              d_contrib=dc, n_iter=n_iter, wins=wins, time_s=time.perf_counter() - t0)
        path.write_text(json.dumps(summary, indent=2))
        print(f"{gname}: unseen {b['unseen']['wbmae']:.2f} seen {b['seen']['wbmae']:.2f} dual {float(b['macro']['wbmae']):.2f} "
              f"Δ {delta:+.2f} better {nb}/10 | Δ contrib " + " ".join(f"{x:+.2f}" for x in dc) +
              f" | n_iter {n_iter} | {'WINS' if wins else 'no'} ({summary[gname]['time_s']:.0f} s)", flush=True)
        print(f"    folds Δ unseen {np.round(du, 2).tolist()} seen {np.round(ds, 2).tolist()}", flush=True)
