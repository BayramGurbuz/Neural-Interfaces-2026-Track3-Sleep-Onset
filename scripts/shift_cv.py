"""Shifted dual CV (16 Nov, Work B selection): the held-out recordings of every fold are streamed from a sampled start window
k instead of the recording start (elapsed time, causal features, detector summaries, baselines and recurrent / TCN states all
restart there; only windows >= k scored, official per-recording W-bMAE). Fold models are retrained exactly as in the dual CV
(same training rows, crop copies, out-of-fold detector outputs, seeds, deterministic settings) at the number of trees / epoch
the CV chose for that fold; then the held-out recordings are predicted under each shift distribution.

Parts (run separately, CPU / GPU in parallel; outputs keyed by distribution set):
  hgb             HGB(c) 4 members, closeness + crop table (ens_crop_cv.py c), n_iter per fold from its npz
  seq NAME        sequence model NAME (SEQ_SPECS), epochs per fold from OUT/b2_seq.json (or the fixed final epochs)
  mix ROWS        mixes "hgb:name+name+..." ; alpha nested LOFO 0.3-0.8 per distribution; Δ vs the first row
Distributions (--dists=, comma list): k0 (no shift, check against the k=0 CV), kK (fixed K windows), UK (uniform 0..K),
  mPUK (P no shift, 1-P uniform 0..K, e.g. m0.5U24), wide (uniform 0..(onset - 60 s) / 5). --draws=J starts per recording
  (J pseudo-recordings), seed from (protocol, fold, draw).

Work C (17 Nov): `hgb --det=ctx` = HGB(c) with the context detector's closeness (ens_crop_cv.py ctx6) -> shiftcv<T>_hgbctx_<proto>.npz,
  mix rows "ctx:..."; seq uw_n10_ctx6_crop.
Work E (18 Nov): `hgb --det=ctx --crop=E` (2 crop copies per recording from U[2,5] / U[0,10] min + ~10 % original rows) -> shiftcv<T>_hgbctxE_<proto>.npz,
  mix rows "ctxE:..."; seq uw_n10_ctx6_cropE, tcn_n10_cropE, cnn3_emb_cropE (crop mix 10 / 50 / 40 %). Same shift samples as the n2c rows (seed from protocol, fold, draw, distribution).
  Approximation: the outer-fold context detectors are not saved, so a shifted stream's first 5 windows keep the closeness computed
  with the pre-shift context (the stream-from-window-2 check moved Spearman by 0.002); exact in the finals and the 40-test runs.

Work D/F (18 Nov): distributions Wa-b (uniform windows a..b) and Fk (fixed k) DROP a recording with no window left (the other
  distributions clamp to the last window); part `fmix` = Work F mixes w * ctx_c2 + (1 - w) * ctx_c3 -> OUT/shiftcv<T>_fmix.json.

Usage: python scripts/shift_cv.py hgb|seq NAME|mix ROWS --dists=k0,U24 [--draws=2] [--tag=T] [--det=ctx]
  -> OUT/shiftcv<T>_hgb_<proto>.npz, OUT/shiftcv<T>_seq_<NAME>_<proto>.npz, OUT/shiftcv<T>_mix.json
"""
import json
import sys
import time

import numpy as np

ARG = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
PART = POS[0]
if PART == "seq":
    from repro import set_deterministic
    set_deterministic()

import torch  # noqa: E402

import hgb_crop_precheck as pc  # noqa: E402
from decision_layer import FINE, evaluate, old_slot_weights, predict, rec_onsets, select  # noqa: E402
from dist_hgb import VARIANTS, class_stats  # noqa: E402
from dual_cv import OUT, seen_splits, unseen_splits  # noqa: E402
from ensemble_dist import MEMBERS, grid_omega, to_grid  # noqa: E402
from gru_cv import ARCHS, CONFIGS, DEV, load_inputs, pad, run_fold, seq_inputs  # noqa: E402
from n2det_feats import MC_SEEDS, summaries  # noqa: E402

B = VARIANTS["B"]
DISTS = ARG.get("--dists", "k0").split(",")
DRAWS = int(ARG.get("--draws", "2"))
TAG = ARG.get("--tag", "")
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CROP = (0.5, 0.35, 0.15)
from gru_cv import CROP_E  # noqa: E402  Work E: 10 % none, 50 % U[2,5] min, 40 % U[0,10] min
EPOCHS_FROM = {"uw_n10_ctx6_cropE": "uw_n10_ctx6_crop", "tcn_n10_cropE": "tcn_n10_n2c_crop", "cnn3_emb_cropE": "cnn3_emb_old_crop"}
SEQ_SPECS = {  # name: (cfg, closeness inputs, CNN-embedding inputs, fixed epochs when the CV choice is not on record)
    "uw_n10_n2c_crop": (dict(CONFIGS["uw_n10"], crop=CROP), True, False, None),
    "tcn_n10": (CONFIGS["tcn_n10"], False, False, 10),
    "tcn_n10_n2c_crop": (dict(CONFIGS["tcn_n10"], crop=CROP), True, False, None),
    "cnn3_emb": (dict(CONFIGS["uw"], run_epochs=40, n_seeds=3), False, True, 10),
    "cnn3_emb_old_crop": (dict(CONFIGS["uw"], run_epochs=40, n_seeds=3, crop=CROP), False, True, None),
    "uw_n10_ctx6_crop": (dict(CONFIGS["uw_n10"], crop=CROP), "ctx6", False, None),
    # Work E (18 Nov): the three sequence models with the Work E crop mix (epochs per fold from the analogous crop recipe's CV)
    "uw_n10_ctx6_cropE": (dict(CONFIGS["uw_n10"], crop=CROP_E), "ctx6", False, None),
    "tcn_n10_cropE": (dict(CONFIGS["tcn_n10"], crop=CROP_E), False, False, None),
    "cnn3_emb_cropE": (dict(CONFIGS["uw"], run_epochs=40, n_seeds=3, crop=CROP_E), False, True, None),
}
DET = ARG.get("--det", "mc")

m = (pc.df.split != "test").to_numpy()
assert (np.flatnonzero(m) == np.arange(m.sum())).all()
d = pc.df[m].reset_index(drop=True)
recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
rec_of = np.empty(len(d), int)
for i, r in enumerate(recs):
    rec_of[r] = i
protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
target = d.target.to_numpy()


def sample_k(dist, rng, r):
    n = len(r)
    if dist == "k0":
        k = 0
    elif dist == "wide":
        onset = float(d.stop[r[-1]] + target[r[-1]])
        k = int(rng.integers(0, max(int((onset - 60) // 5), 0) + 1))
    elif dist.startswith("m"):
        p, K = dist[1:].split("U")
        k = 0 if rng.random() < float(p) else int(rng.integers(0, int(K) + 1))
    elif dist.startswith("U"):
        k = int(rng.integers(0, int(dist[1:]) + 1))
    elif dist.startswith("W"):  # Work D (18 Nov): uniform windows a..b; no window left -> recording not delivered (None)
        a, b_ = map(int, dist[1:].split("-"))
        k = int(rng.integers(a, b_ + 1))
        return k if k <= n - 1 else None
    elif dist.startswith("F"):  # fixed k, drop convention
        k = int(dist[1:])
        return k if k <= n - 1 else None
    else:
        k = int(dist[1:])
    return min(k, n - 1)


def shifted(proto, k_fold, dist):
    """-> list of (recording index, start k, draw j) for the held-out recordings of the fold (deterministic)."""
    te = protocols[proto][k_fold][1]
    out = []
    for j in range(1 if dist == "k0" else DRAWS):
        rng = np.random.default_rng([sum(map(ord, proto)), k_fold, j, sum(map(ord, dist))])  # no hash(): randomised per process
        for i in np.unique(rec_of[te]):
            kk = sample_k(dist, rng, recs[i])
            if kk is not None:
                out.append((int(i), kk, j))
    return out


def close_values(proto, k, src="multi"):
    return np.mean([np.load(OUT / f"n2det_{src}_s{s}_{proto}_f{k}.npz")["close"] for s in MC_SEEDS], axis=0)


def part_hgb():
    from cv_utils import apply_bin_edges, fit_bin_edges
    from sklearn.ensemble import HistGradientBoostingClassifier
    from dist_hgb import to_class
    X0 = pc.X817[m]
    CROPH = ARG.get("--crop")
    if CROPH == "E":  # Work E (18 Nov): 2 copies per recording, k ~ U[2,5] min and U[0,10] min (empty copy dropped), all features
        # re-streamed from the crop (closeness summaries per copy below); original rows subsampled to ~10 % of the table
        rng = np.random.default_rng(0)
        crop_recs = []
        for r in recs:
            for a, b_ in ((24, 60), (0, 120)):
                kk = int(rng.integers(a, b_ + 1))
                if kk <= len(r) - 1:
                    crop_recs.append(r[kk:])
        c_rows = np.concatenate(crop_recs)
        c_X = np.concatenate([pc.restream(rr) for rr in crop_recs])
        keep_o = rng.random(len(d)) < len(c_rows) / 9 / len(d)
        print(f"Work E table: {len(crop_recs)} crop copies, {len(c_rows)} crop rows, {int(keep_o.sum())} original rows "
              f"({keep_o.sum() / (keep_o.sum() + len(c_rows)):.3f} of the table)", flush=True)
    else:
        c_rows, c_X, ks = pc.cropped_copies(recs, np.random.default_rng(0))
        crop_recs = [r[k:] for r, k in zip(recs, ks)]
        keep_o = np.ones(len(d), bool)
    tl_c = d.timeline.to_numpy()[c_rows]
    y_ext = np.concatenate([target, target[c_rows]])
    fs = "n2ctx6_crop" if DET == "ctx" else "n2c_crop"
    iters = {nm: np.load(OUT / f"ens_dyn_maxmin_spec_eye_{fs}_{nm}.npz", allow_pickle=True)["n_iter"].item() for nm in MEMBERS}
    probas = {nm: np.load(OUT / f"ens_dyn_maxmin_spec_eye_{fs}_{nm}.npz", allow_pickle=True)["probas"].item() for nm in MEMBERS}
    for proto, splits in protocols.items():
        path = OUT / f"shiftcv{TAG}_hgb{'ctx' if DET == 'ctx' else ''}{CROPH or ''}_{proto}.npz"
        if path.exists():
            print(f"hgb {proto}: cached", flush=True)
            continue
        res = {}
        for k, (tr, te) in enumerate(splits):
            t0 = time.perf_counter()
            v = close_values(proto, k, "ctx6" if DET == "ctx" else "multi")
            S = summaries(v, recs)
            Sc = (np.concatenate([summaries(v[rr], [np.arange(len(rr))]) for rr in crop_recs]) if CROPH == "E"
                  else summaries(v, crop_recs)[c_rows])
            Xe = np.concatenate([np.concatenate([X0, S], 1), np.concatenate([c_X, Sc], 1)])
            tr_e = np.concatenate([tr & keep_o, np.isin(tl_c, np.unique(d.timeline.to_numpy()[tr]))])
            bins = fit_bin_edges(Xe[tr_e])
            Xb = apply_bin_edges(Xe[tr_e], bins)
            clfs = []
            for nm, (edges, params, _) in MEMBERS.items():
                clf = HistGradientBoostingClassifier(max_iter=int(iters[nm][proto][k]), max_features=0.5, early_stopping=False,
                                                     random_state=0, **params)
                clfs.append((clf.fit(Xb, to_class(y_ext[tr_e], edges)), edges, nm))
            # check: unshifted held-out probabilities reproduce the CV's
            P0 = clfs[0][0].predict_proba(apply_bin_edges(Xe[:len(d)][te], bins))
            dev = float(np.abs(P0 - probas[clfs[0][2]][proto][k]).max()) if CROPH is None else float("nan")
            for dist in DISTS:
                Ph = []
                for i, kk, j in shifted(proto, k, dist):
                    rows = recs[i][kk:]
                    H = np.concatenate([pc.restream(rows), summaries(v[rows], [np.arange(len(rows))])], 1)
                    Hb = apply_bin_edges(H, bins)
                    Ph.append(np.mean([to_grid(c.predict_proba(Hb), e) for c, e, _ in clfs], axis=0).astype(np.float32))
                res[f"{dist}_f{k}"] = np.concatenate(Ph)
            print(f"hgb {proto} fold {k}: refit + {len(DISTS)} distributions in {time.perf_counter() - t0:.0f} s; unshifted vs CV probas "
                  f"max |Δ| {dev:.2e}", flush=True)
        np.savez(path, **res)


def part_seq(name):
    cfg, n2c, emb, fixed = SEQ_SPECS[name]
    d2, recs2, F, gain = load_inputs()
    assert (d2.timeline.to_numpy() == d.timeline.to_numpy()).all() and (d2.stop.to_numpy() == d.stop.to_numpy()).all()
    cv = json.loads((OUT / "b2_seq.json").read_text()).get(EPOCHS_FROM.get(name, name))
    n_seeds = cfg.get("n_seeds", 10)
    for proto, splits in protocols.items():
        path = OUT / f"shiftcv{TAG}_seq_{name}_{proto}.npz"
        if path.exists():
            print(f"{name} {proto}: cached", flush=True)
            continue
        res = {}
        for k, (tr, te) in enumerate(splits):
            t0 = time.perf_counter()
            ep = int(cv[proto]["epochs"][k]) if cv else fixed
            v = (close_values(proto, k, "ctx6" if n2c == "ctx6" else "multi") if n2c else None)
            models = []
            for s in range(n_seeds):
                E = np.load(OUT / f"cnn_cnn3_{proto}_f{k}_s{s}.npz")["emb"].astype(np.float32) if emb else None
                if n2c:
                    def extra(r, v=v):
                        return summaries(v[r], [np.arange(len(r))])
                else:
                    extra = E
                net, mu, sd = run_fold(d2, recs2, F, gain, tr, None, cfg, s, stop_epoch=ep, extra=extra)
                models.append((net, mu, sd, extra))
            for dist in DISTS:
                items = shifted(proto, k, dist)
                P = []
                for net, mu, sd, extra in models:
                    xs = []
                    for i, kk, j in items:
                        rows = recs2[i][kk:]
                        ex = None if extra is None else (extra(rows) if callable(extra) else extra[rows])
                        xs.append((seq_inputs(F[rows], ex) - mu) / sd)
                    with torch.no_grad():
                        out = torch.softmax(net(torch.tensor(pad(xs), device=DEV))[0], -1).cpu().numpy()
                    P.append(np.concatenate([out[n, :len(x)] for n, x in enumerate(xs)]))
                res[f"{dist}_f{k}"] = np.mean(P, axis=0).astype(np.float32)
            print(f"{name} {proto} fold {k}: epoch {ep}, {n_seeds} seeds + {len(DISTS)} distributions in {time.perf_counter() - t0:.0f} s",
                  flush=True)
        np.savez(path, **res)


def part_mix(rows_arg):
    import pandas as pd
    onset = rec_onsets(d)
    rows = [(r.split(":")[0], r.split(":")[1].split("+")) for r in rows_arg.split(",")]
    H = {"n2c": {p: np.load(OUT / f"shiftcv{TAG}_hgb_{p}.npz") for p in protocols}} if (OUT / f"shiftcv{TAG}_hgb_unseen.npz").exists() else {}
    if any(r[0] == "ctx" for r in rows):
        H["ctx"] = {p: np.load(OUT / f"shiftcv{TAG}_hgbctx_{p}.npz") for p in protocols}
    if any(r[0] == "ctxE" for r in rows):  # Work E HGB (hgb --det=ctx --crop=E)
        H["ctxE"] = {p: np.load(OUT / f"shiftcv{TAG}_hgbctxE_{p}.npz") for p in protocols}
    out = {}
    for dist in DISTS:
        E = {}
        for hg, seqs in rows:
            assert hg in H, "HGB sets retrained here: n2c, ctx, ctxE"
            Z = {s: {p: np.load(OUT / f"shiftcv{TAG}_seq_{s}_{p}.npz") for p in protocols} for s in seqs}
            folds = []
            for proto, splits in protocols.items():
                for k, (tr, _) in enumerate(splits):
                    items = shifted(proto, k, dist)
                    idx = np.concatenate([recs[i][kk:] for i, kk, _ in items])
                    dte = d.iloc[idx].reset_index(drop=True).copy()
                    dte["timeline"] = np.concatenate([[f"{d.timeline[recs[i][0]]}#{j}"] * (len(recs[i]) - kk) for i, kk, j in items])
                    dtr = d[tr].reset_index(drop=True)
                    _, ob = class_stats(dtr, B)
                    folds.append(dict(proto=proto, k=k, dte=dte, t=dte.stop.to_numpy().astype(np.float64),
                                      seq=[to_grid(Z[s][proto][f"{dist}_f{k}"].astype(np.float64), B) @ FINE for s in seqs],
                                      hgb=H[hg][proto][f"{dist}_f{k}"].astype(np.float64) @ FINE,
                                      w_old=old_slot_weights(ob, grid_omega(dtr)),
                                      D=onset.loc[d.timeline.to_numpy()[idx]].to_numpy()))
            c0 = {a: [predict(f, "old", 0.0, alpha=a) for f in folds] for a in ALPHAS}
            p, ch, al = select(folds, c0)
            E[f"{hg}:{'+'.join(seqs)}"] = (evaluate(folds, p), ch, al)
        r0 = list(E.values())[0][0]
        res = {}
        for key, (e, ch, al) in E.items():
            du = np.array(e["unseen"]["folds"]) - np.array(r0["unseen"]["folds"])
            ds = np.array(e["seen"]["folds"]) - np.array(r0["seen"]["folds"])
            res[key] = dict(unseen=e["unseen"]["mean"], seen=e["seen"]["mean"], macro=e["macro"]["mean"],
                            delta=e["macro"]["mean"] - r0["macro"]["mean"], better=int((du < -1e-9).sum() + (ds < -1e-9).sum()),
                            d_unseen=du.tolist(), d_seen=ds.tolist(), contrib=e["macro"]["contrib"], alpha=ch, alpha_all=al)
        out[dist] = res
        print(f"\n== {dist} (Δ vs the first row)")
        for key, v in res.items():
            print(f"  {key:62s} unseen {v['unseen']:6.2f} seen {v['seen']:6.2f} dual {v['macro']:6.2f} Δ {v['delta']:+6.2f} "
                  f"{v['better']:2d}/10 α {v['alpha_all']} | folds Δ {np.round(v['d_unseen'], 2).tolist()} {np.round(v['d_seen'], 2).tolist()}",
                  flush=True)
    path = OUT / ARG.get("--mixout", f"shiftcv{TAG}_mix.json")  # 17 Nov: separate file per row set (Work C: shiftcv_mix_wc.json)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=2, default=float))


def part_fmix():
    """Work F (18 Nov): w * ctx_c2 + (1 - w) * ctx_c3 pre-decision mixes (shared HGB ctx and GRU ctx; ctx_c2 adds TCN and GRU-emb),
    alpha fixed at the packages' values (0.5 / 0.4); w nested LOFO over 0..1 (0.1 steps); Δ vs w = 0 (ctx_c3)."""
    from decision_layer import slot_median, to_slots
    onset = rec_onsets(d)
    H = {p: np.load(OUT / f"shiftcv{TAG}_hgbctx_{p}.npz") for p in protocols}
    names = ["uw_n10_ctx6_crop", "tcn_n10", "cnn3_emb"]
    Z = {s: {p: np.load(OUT / f"shiftcv{TAG}_seq_{s}_{p}.npz") for p in protocols} for s in names}
    W_GRID = [round(0.1 * i, 1) for i in range(11)]

    def pred_w(f, w, chunk=4096):
        c_h, c_g, c_o = 0.5 * w + 0.6 * (1 - w), w * 0.5 / 3 + 0.4 * (1 - w), w * 0.5 / 3
        out = np.empty(len(f["t"]))
        ws, wh = f["w_old"]
        for i0 in range(0, len(out), chunk):
            sl = slice(i0, i0 + chunk)
            t = f["t"][sl]
            q = [(x * ws) / (x * ws).sum(1, keepdims=True) for x in (to_slots(y[sl], t, 0.0) for y in f["seq"])]
            h = to_slots(f["hgb"][sl], t, 0.0) * wh
            M = c_h * h / h.sum(1, keepdims=True) + c_g * q[0] + c_o * (q[1] + q[2])
            out[sl] = slot_median(M)
        return out

    out = {}
    for dist in DISTS:
        folds = []
        for proto, splits in protocols.items():
            for k, (tr, _) in enumerate(splits):
                items = shifted(proto, k, dist)
                idx = np.concatenate([recs[i][kk:] for i, kk, _ in items])
                dte = d.iloc[idx].reset_index(drop=True).copy()
                dte["timeline"] = np.concatenate([[f"{d.timeline[recs[i][0]]}#{j}"] * (len(recs[i]) - kk) for i, kk, j in items])
                dtr = d[tr].reset_index(drop=True)
                _, ob = class_stats(dtr, B)
                folds.append(dict(proto=proto, k=k, dte=dte, t=dte.stop.to_numpy().astype(np.float64),
                                  seq=[to_grid(Z[s][proto][f"{dist}_f{k}"].astype(np.float64), B) @ FINE for s in names],
                                  hgb=H[proto][f"{dist}_f{k}"].astype(np.float64) @ FINE,
                                  w_old=old_slot_weights(ob, grid_omega(dtr)), D=onset.loc[d.timeline.to_numpy()[idx]].to_numpy()))
        cand = {w: [pred_w(f, w) for f in folds] for w in W_GRID}
        res = {}
        for w in W_GRID:  # fixed-w rows
            e = evaluate(folds, cand[w])
            res[f"w={w}"] = e
        p, ch, wa = select(folds, cand)
        res["nested"] = evaluate(folds, p)
        r0 = res["w=0.0"]
        rows = {}
        for key, e in res.items():
            du = np.array(e["unseen"]["folds"]) - np.array(r0["unseen"]["folds"])
            ds = np.array(e["seen"]["folds"]) - np.array(r0["seen"]["folds"])
            rows[key] = dict(unseen=e["unseen"]["mean"], seen=e["seen"]["mean"], macro=e["macro"]["mean"],
                             delta=e["macro"]["mean"] - r0["macro"]["mean"], better=int((du < -1e-9).sum() + (ds < -1e-9).sum()),
                             d_unseen=du.tolist(), d_seen=ds.tolist(), contrib=e["macro"]["contrib"])
        rows["nested"].update(w_folds=ch, w_all=wa)
        out[dist] = rows
        print(f"\n== {dist} (Δ vs w = 0, ctx_c3; w = 1 is ctx_c2); nested w per fold {ch}, all-fold {wa}")
        for key, v in rows.items():
            print(f"  {key:8s} unseen {v['unseen']:6.2f} seen {v['seen']:6.2f} dual {v['macro']:6.2f} Δ {v['delta']:+6.2f} {v['better']:2d}/10 | "
                  f"folds Δ {np.round(v['d_unseen'], 2).tolist()} {np.round(v['d_seen'], 2).tolist()}", flush=True)
    path = OUT / f"shiftcv{TAG}_fmix.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=2, default=float))


if __name__ == "__main__":
    t0 = time.perf_counter()
    {"hgb": lambda: part_hgb(), "seq": lambda: part_seq(POS[1]), "mix": lambda: part_mix(POS[1]), "fmix": lambda: part_fmix()}[PART]()
    print(f"{PART} done ({time.perf_counter() - t0:.0f} s)")
