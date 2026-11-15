"""B2: mix evaluation of new HGB(c) / sequence-model probabilities with the pi(D) grid decision (no training).

Combination = (HGB(c) feature set, sequence-model set). Each is decided like mix3_540_gridpi (b1_grid_pi.predict_pi:
pi(D) bins4 from the fold's training recordings, class weights omega, structure "comp"), with alpha chosen nested-LOFO
from ALPHAS, and also without the grid (pi = 0, decision_layer.predict) to see whether the c-group gain adds to the
grid decision or overlaps it. Reference = mix3 + pi(D) (38.27, saved b1pi_bins4 preds; reproduced here).
Scores: official binned_mae; Δ fold-paired vs the reference; contributions; on-grid / off-grid recordings.

Usage: python scripts/b2_mix.py [ROWS TAG [nogrid]] [--ref2=HGB:SEQ]   (writes OUT/b2_mix[_TAG].json, OUT/b2mix_<row>_{unseen,seen}_preds.npz)
  nogrid (14 Nov rule change: the 30 s onset grid is banned): only the ungridded decision is evaluated and the
  reference is the ungridded mix3 recipe (HGB(old) + seq(old), alpha 0.7 fixed = 39.40, sealed 37.79).
  ROWS = comma-separated hgb:seq pairs, e.g. "a+c:old,a+c:a"; the reference row (old, old) is always added.
"""
import gc
import json
import sys

import numpy as np

from b1_grid_pi import pi_table, predict_pi
from decision_layer import FINE, by_proto, evaluate, old_slot_weights, predict, rec_onsets, select
from dist_hgb import VARIANTS, class_stats
from dual_cv import OUT, load_trainval, seen_splits, unseen_splits
from ensemble_dist import GRID, MEMBERS, grid_omega, to_grid
from range_breakdown import load_preds

B = VARIANTS["B"]
HGB_MEMBERS = ["B", "B_h1", "B_h2", "B2"]
SEQ_OLD = ["uw_n10", "tcn_n10", "cnn3_emb"]
SEQ = {"old": SEQ_OLD, "a": [f"{s}_bip" for s in SEQ_OLD], "a+c": [f"{s}_bip_epoch" for s in SEQ_OLD],
       "n": [f"{s}_n2det" for s in SEQ_OLD],  # n = CNN N2-detector summaries (Work 2)
       "nG": ["uw_n10_n2det", "tcn_n10", "cnn3_emb"],  # only the GRU with n2det, TCN / GRU-emb old
       "crop": [f"{s}_old_crop" for s in SEQ_OLD],  # stream-start crop augmentation (14 Nov), old inputs
       "cropG": ["uw_n10_old_crop", "tcn_n10", "cnn3_emb"],  # only the GRU crop-augmented
       "cropGc": ["uw_n10_n2c_crop", "tcn_n10", "cnn3_emb"]}  # GRU crop-augmented + multi-class closeness inputs (15 Nov)
HGB = {"old": "dyn_maxmin_spec_eye", "a": "dyn_maxmin_spec_eye_bip", "a+c": "dyn_maxmin_spec_eye_bip+epoch",
       "n": "dyn_maxmin_spec_eye_n2det", "nB": "dyn_maxmin_spec_eye_n2Bonly",
       "n2c": "dyn_maxmin_spec_eye_n2c_crop"}  # 15 Nov: 4 members + multi-class closeness + crop-augmented table (ens_crop_cv.py c)  # nB: only the B member with n2det (indicator)
ROWS = [("old", "old"), ("a", "a"), ("a", "a+c"), ("old", "a"), ("a", "old"), ("old", "a+c")]
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # nested-LOFO grid (14 Nov: 0.3-0.8, HGB may now beat the sequence models)
CUTS = [900.0, 1100.0, 1300.0]


def build(d, protocols, hgb_set, seq_names):
    hgb = {m: np.load(OUT / f"ens_{hgb_set}_{m}.npz", allow_pickle=True)["probas"].item() for m in HGB_MEMBERS}
    onset = rec_onsets(d)
    folds = []
    for proto, splits in protocols.items():
        zs = [np.load(OUT / f"gru_{n}_{proto}_proba.npz") for n in seq_names]
        for k, (tr, te) in enumerate(splits):
            dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
            _, ob = class_stats(dtr, B)
            D_tr = onset.loc[dtr.timeline.unique()].to_numpy()
            folds.append(dict(
                proto=proto, k=k, dte=dte, t=dte.stop.to_numpy().astype(np.float64),
                seq=[to_grid(z[f"arr_{k}"].astype(np.float64), B) @ FINE for z in zs],
                hgb=np.mean([to_grid(hgb[m][proto][k], MEMBERS[m][0]) for m in HGB_MEMBERS], axis=0) @ FINE,
                w_old=old_slot_weights(ob, grid_omega(dtr)), tab=pi_table(D_tr, CUTS)[0],
                D=onset.loc[dte.timeline.to_numpy()].to_numpy()))
    return folds


def summarise(e, r0):
    du = np.array(e["unseen"]["folds"]) - np.array(r0["unseen"]["folds"])
    ds = np.array(e["seen"]["folds"]) - np.array(r0["seen"]["folds"])
    return dict(unseen=e["unseen"]["mean"], seen=e["seen"]["mean"], macro=e["macro"]["mean"],
                delta=e["macro"]["mean"] - r0["macro"]["mean"], better=int((du < -1e-9).sum() + (ds < -1e-9).sum()),
                d_unseen=du.tolist(), d_seen=ds.tolist(), contrib=e["macro"]["contrib"],
                d_contrib=(np.array(e["macro"]["contrib"]) - np.array(r0["macro"]["contrib"])).tolist(),
                on_grid=e["macro"]["on_grid"], off_grid=e["macro"]["off_grid"],
                on_unseen=e["unseen"]["on_grid"], on_seen=e["seen"]["on_grid"],
                off_unseen=e["unseen"]["off_grid"], off_seen=e["seen"]["off_grid"])


if __name__ == "__main__":
    TAG = ""
    NOGRID = len(sys.argv) > 3 and sys.argv[3] == "nogrid"
    if len(sys.argv) > 1:
        ROWS = [("old", "old")] + [tuple(x.split(":")) for x in sys.argv[1].split(",")]
        TAG = "_" + sys.argv[2]
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    res, E = {}, {}
    ref_saved = load_preds("b1pi_bins4", protocols)
    for hg, sq in ROWS:
        key = f"HGB({hg}) + seq({sq})"
        folds = build(d, protocols, HGB[hg], SEQ[sq])
        c0 = {a: [predict(f, "old", 0.0, alpha=a) for f in folds] for a in ALPHAS}
        p_nog, ch_n, all_n = select(folds, c0)
        if NOGRID:
            if (hg, sq) == ("old", "old"):
                E["ref"] = evaluate(folds, c0[0.7])
                print(f"reference = ungridded mix3, alpha 0.7: macro {E['ref']['macro']['mean']:.2f} (expected 39.40)", flush=True)
            E[key + " nogrid"] = evaluate(folds, p_nog)
            res[key] = dict(alpha_nogrid=dict(folds=ch_n, all=all_n))
            P = by_proto(folds, p_nog)
            for pr in P:
                np.savez(OUT / f"b2mix_{hg}_{sq}_nogrid_{pr}_preds.npz".replace("+", "_"), *P[pr])
            print(f"  done {key}: alpha no grid {ch_n} (all {all_n})", flush=True)
            del folds, c0
            gc.collect()
            continue
        cg = {a: [predict_pi(f, f["tab"], alpha=a) for f in folds] for a in ALPHAS}
        p_grid, ch_g, all_g = select(folds, cg)
        if (hg, sq) == ("old", "old"):
            dev = max(float(np.abs(x - ref_saved[f["proto"]][f["k"]]).max()) for x, f in zip(cg[0.7], folds))
            print(f"reference (alpha 0.7, pi(D)) reproduced: max |Δ| vs saved b1pi_bins4 = {dev:.2e} s", flush=True)
            E["ref"] = evaluate(folds, [ref_saved[f["proto"]][f["k"]] for f in folds])
        E[key + " grid"] = evaluate(folds, p_grid)
        E[key + " nogrid"] = evaluate(folds, p_nog)
        res[key] = dict(alpha_grid=dict(folds=ch_g, all=all_g), alpha_nogrid=dict(folds=ch_n, all=all_n))
        for tag, p in (("grid", p_grid), ("nogrid", p_nog)):
            P = by_proto(folds, p)
            name = f"b2mix_{hg}_{sq}_{tag}".replace("+", "_")
            for pr in P:
                np.savez(OUT / f"{name}_{pr}_preds.npz", *P[pr])
        print(f"  done {key}: alpha grid {ch_g} (all {all_g}), no grid {ch_n} (all {all_n})", flush=True)
        del folds, cg, c0
        gc.collect()
    r0 = E["ref"]
    rows = {k: summarise(e, r0) for k, e in E.items()}
    REF2 = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ref2=")), None)  # e.g. n:cropG (candidate 1, 38.86)
    rows2 = {}
    if REF2:
        hg2, sq2 = REF2.split(":")
        r2 = E[f"HGB({hg2}) + seq({sq2}) nogrid"]
        rows2 = {k: summarise(e, r2) for k, e in E.items() if k != "ref"}
        res["ref2"] = dict(row=REF2, macro=r2["macro"]["mean"])
    for k, v in rows.items():
        base = k.rsplit(" ", 1)[0]
        if base in res:
            res[base][k.rsplit(" ", 1)[1]] = v
    res["ref"] = rows["ref"]
    for k, v in rows2.items():
        base = k.rsplit(" ", 1)[0]
        if base in res:
            res[base][k.rsplit(" ", 1)[1] + "_vs_ref2"] = v
    print(f"\n{'row':34s} unseen   seen  macro      Δ better | Δ contrib [0,40) [40,90) [90,300) [300,600] | on-grid  off-grid")
    for k, v in rows.items():
        print(f"{k:34s} {v['unseen']:6.2f} {v['seen']:6.2f} {v['macro']:6.2f} {v['delta']:+6.2f} {v['better']:2d}/10 | "
              + " ".join(f"{x:+8.2f}" for x in v["d_contrib"]) + f" | {v['on_grid']:6.2f}  {v['off_grid']:6.2f}")
        print(f"{'':34s} folds Δ unseen {np.round(v['d_unseen'], 2).tolist()} seen {np.round(v['d_seen'], 2).tolist()}")
    if rows2:
        print(f"\nΔ vs ref2 = HGB({hg2}) + seq({sq2}) nogrid ({res['ref2']['macro']:.2f}):")
        for k, v in rows2.items():
            print(f"{k:34s} {v['macro']:6.2f} {v['delta']:+6.2f} {v['better']:2d}/10 | " + " ".join(f"{x:+8.2f}" for x in v["d_contrib"])
                  + f" | folds Δ unseen {np.round(v['d_unseen'], 2).tolist()} seen {np.round(v['d_seen'], 2).tolist()}")
    # c effect: (a + c) minus (a), with and without the grid decision, on-grid / off-grid recordings
    print("\nc-group effect (sequence inputs a+c minus a), macro W-bMAE differences:")
    c_eff = {}
    for hg in ("a", "old"):
        for tag in ("grid", "nogrid"):
            if f"HGB({hg}) + seq(a+c) {tag}" not in rows or f"HGB({hg}) + seq(a) {tag}" not in rows:
                continue
            x, y = rows[f"HGB({hg}) + seq(a+c) {tag}"], rows[f"HGB({hg}) + seq(a) {tag}"]
            c_eff[f"HGB({hg}) {tag}"] = dict(all=x["macro"] - y["macro"], on_grid=x["on_grid"] - y["on_grid"],
                                             off_grid=x["off_grid"] - y["off_grid"],
                                             better=int(sum(np.array(x["d_unseen"]) < np.array(y["d_unseen"]))
                                                        + sum(np.array(x["d_seen"]) < np.array(y["d_seen"]))))
            v = c_eff[f"HGB({hg}) {tag}"]
            print(f"  HGB({hg}), {tag:6s}: all {v['all']:+.2f}  on-grid {v['on_grid']:+.2f}  off-grid {v['off_grid']:+.2f}"
                  f"  (better in {v['better']}/10 folds)")
    res["c_effect"] = c_eff
    (OUT / f"b2_mix{TAG}.json").write_text(json.dumps(res, indent=2, default=str))
