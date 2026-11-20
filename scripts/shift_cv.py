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

Work G (19 Nov): `hgb ... --members=B --seeds=0,1,2` refits only those members, once per seed -> keys <dist>_f<k>_s<seed> in
  shiftcv<T>_hgbctx<crop>_<members>_s<seeds>_<proto>.npz; part `seedeval` (same flags) = HGB-only decision, single seed vs seed mean.

Work I (19 Nov): part `imix` = fixed-weight package mixes (M = ctx_c3 / 2 + ctxE_c2hs3p / 2, M', Work J rows), no selection, needs the
  Work I HGB E runs (scripts/worki_chain.sh) -> OUT/shiftcv_i_mix.json.

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
if PART in ("seq", "e2e"):
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
    memb = ARG["--members"].split(",") if "--members" in ARG else list(MEMBERS)  # Work G (19 Nov): member subset and seeds
    seeds = [int(x) for x in ARG["--seeds"].split(",")] if "--seeds" in ARG else [0]
    SEED_SFX = f"_{'+'.join(memb)}_s{''.join(map(str, seeds))}" if "--seeds" in ARG else ""
    for proto, splits in protocols.items():
        path = OUT / f"shiftcv{TAG}_hgb{'ctx' if DET == 'ctx' else ''}{CROPH or ''}{SEED_SFX}_{proto}.npz"
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
                if nm not in memb:
                    continue
                for sd_ in seeds:
                    clf = HistGradientBoostingClassifier(max_iter=int(iters[nm][proto][k]), max_features=0.5, early_stopping=False,
                                                         random_state=sd_, **params)
                    clfs.append((clf.fit(Xb, to_class(y_ext[tr_e], edges)), edges, nm, sd_))
            # check: unshifted held-out probabilities reproduce the CV's
            P0 = clfs[0][0].predict_proba(apply_bin_edges(Xe[:len(d)][te], bins))
            dev = float(np.abs(P0 - probas[clfs[0][2]][proto][k]).max()) if CROPH is None else float("nan")
            for dist in DISTS:
                Ph = []
                for i, kk, j in shifted(proto, k, dist):
                    rows = recs[i][kk:]
                    H = np.concatenate([pc.restream(rows), summaries(v[rows], [np.arange(len(rows))])], 1)
                    Hb = apply_bin_edges(H, bins)
                    if SEED_SFX:  # per seed: mean over the selected members
                        Ph.append(np.stack([np.mean([to_grid(c.predict_proba(Hb), e) for c, e, _, s_ in clfs if s_ == sd_], axis=0)
                                            for sd_ in seeds]).astype(np.float32))
                    else:
                        Ph.append(np.mean([to_grid(c.predict_proba(Hb), e) for c, e, _, _ in clfs], axis=0).astype(np.float32))
                if SEED_SFX:
                    for si, sd_ in enumerate(seeds):
                        res[f"{dist}_f{k}_s{sd_}"] = np.concatenate([p[si] for p in Ph])
                else:
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


IMIX_SEQ = ["uw_n10_ctx6_crop", "tcn_n10", "cnn3_emb"]
_G3 = 0.4 / 3
IMIX_PK = {  # (HGB ctx, HGB E s0, HGB E s3p, GRU ctx, TCN, GRU-emb)
    "ctx_c3": (0.6, 0, 0, 0.4, 0, 0), "ctx_c2": (0.5, 0, 0, 0.5 / 3, 0.5 / 3, 0.5 / 3),
    "ctxE_c2h": (0, 0.6, 0, _G3, _G3, _G3), "ctxE_c2hs3p": (0, 0, 0.6, _G3, _G3, _G3)}


def imix_folds(dist, lazy=False):
    """Work I fold inputs under one distribution (factored out of part_imix on 20 Nov, unchanged): per fold the held-out stream table, the
    fine (601-cell) distributions of the 3 HGB sets and 3 sequence components, the old slot weights. Returns (folds, E composition max dev)."""
    onset = rec_onsets(d)
    t = "" if dist == "k0" else "_d"
    R1 = {p: np.load(OUT / f"shiftcv_i_hgbctxE_B+B_h2_s012_{p}.npz") for p in protocols}
    R2 = {p: np.load(OUT / f"shiftcv_i_hgbctxE_B_h1+B2_s0_{p}.npz") for p in protocols}
    Ed = {p: np.load(OUT / f"shiftcv_d_hgbctxE_{p}.npz") for p in protocols}
    src = {"hctx": {p: np.load(OUT / f"shiftcv{t}_hgbctx_{p}.npz") for p in protocols},
           **{s: {p: np.load(OUT / f"shiftcv{t}_seq_{s}_{p}.npz") for p in protocols} for s in IMIX_SEQ}}
    folds, dev = [], 0.0
    for proto, splits in protocols.items():
        for k, (tr, _) in enumerate(splits):
            r1 = [R1[proto][f"{dist}_f{k}_s{s}"].astype(np.float64) for s in (0, 1, 2)]
            if dist == "k0":
                e0 = 0.5 * r1[0] + 0.5 * R2[proto][f"k0_f{k}_s0"].astype(np.float64)
            else:
                e0 = Ed[proto][f"{dist}_f{k}"].astype(np.float64)
                if dist == "W24-60":  # composition check
                    dev = max(dev, float(np.abs(0.5 * r1[0] + 0.5 * R2[proto][f"W24-60_f{k}_s0"] - e0).max()))
            es3 = e0 + 0.5 * (np.mean(r1, axis=0) - r1[0])
            items = shifted(proto, k, dist)
            idx = np.concatenate([recs[i][kk:] for i, kk, _ in items])
            dte = d.iloc[idx].reset_index(drop=True).copy()
            dte["timeline"] = np.concatenate([[f"{d.timeline[recs[i][0]]}#{j}"] * (len(recs[i]) - kk) for i, kk, j in items])
            dtr = d[tr].reset_index(drop=True)
            _, ob = class_stats(dtr, B)
            folds.append(dict(proto=proto, k=k, dte=dte, t=dte.stop.to_numpy().astype(np.float64), tr=tr,
                              hgb_g=[src["hctx"][proto][f"{dist}_f{k}"].astype(np.float64), e0, es3],
                              seq_g=[to_grid(src[s][proto][f"{dist}_f{k}"].astype(np.float64), B) for s in IMIX_SEQ],
                              w_old=old_slot_weights(ob, grid_omega(dtr)), D=onset.loc[d.timeline.to_numpy()[idx]].to_numpy()))
            if not lazy:  # 20 Nov: lazy=True keeps only the 25-cell grid arrays (fine 601-cell arrays of all folds need ~10 GB)
                fl = folds[-1]
                fl["hgb"], fl["seq"] = [x @ FINE for x in fl["hgb_g"]], [x @ FINE for x in fl["seq_g"]]
    return folds, dev


def imix_pred(f, c, chunk=4096):
    """Fixed-coefficient package mix of one fold -> point predictions (continuous weighted median of the metric-weighted mix)."""
    from decision_layer import slot_median, to_slots
    out = np.empty(len(f["t"]))
    ws, wh = f["w_old"]
    for i0 in range(0, len(out), chunk):
        sl = slice(i0, i0 + chunk)
        t = f["t"][sl]
        M = 0.0
        for ci, x in zip(c[:3], f["hgb"]):
            if ci:
                h = to_slots(x[sl], t, 0.0) * wh
                M = M + ci * h / h.sum(1, keepdims=True)
        for ci, x in zip(c[3:], f["seq"]):
            if ci:
                q = to_slots(x[sl], t, 0.0) * ws
                M = M + ci * q / q.sum(1, keepdims=True)
        out[sl] = slot_median(M)
    return out


def part_imix():
    """Work I (19 Nov): fixed-weight mixes of whole packages (no selection), pre-decision metric-weighted distributions averaged, then the
    continuous weighted median. Package = coefficients on the normalised HGB sets (ctx; E single seed; E with B and B_h2 x 3 seeds) and the
    normalised sequence components (GRU ctx, TCN, GRU-emb). E pieces: Work E single-seed file under the W distributions; k0 rebuilt from the
    Work I runs (0.5 * mean(B, B_h2) seed 0 + 0.5 * mean(B_h1, B2) seed 0, checked against the Work E file under W24-60);
    E-s3p = E single seed + 0.5 * (seed mean - seed 0) of mean(B, B_h2) (= members weighted 1/12 x3, 1/4, 1/12 x3, 1/4).
    k0 inputs of HGB ctx and the sequence models from the untagged Work C files, W from the Work F files (tag _d)."""
    PK = IMIX_PK
    ROWS = {"ctx_c3": {"ctx_c3": 1.0}, "ctxE_c2hs3p": {"ctxE_c2hs3p": 1.0}, "ctxE_c2h": {"ctxE_c2h": 1.0}, "ctx_c2": {"ctx_c2": 1.0},
            "M = c3/2 + c2hs3p/2": {"ctx_c3": 0.5, "ctxE_c2hs3p": 0.5}, "M' = c3/2 + c2h/2": {"ctx_c3": 0.5, "ctxE_c2h": 0.5},
            "J = (c3 + c2 + c2hs3p)/3": {"ctx_c3": 1 / 3, "ctx_c2": 1 / 3, "ctxE_c2hs3p": 1 / 3},
            "J' = (c3 + c2 + c2h)/3": {"ctx_c3": 1 / 3, "ctx_c2": 1 / 3, "ctxE_c2h": 1 / 3}}
    COEF = {r: np.sum([w * np.asarray(PK[pk]) for pk, w in mix.items()], axis=0) for r, mix in ROWS.items()}
    pred = imix_pred

    out = {}
    for dist in DISTS:
        folds, dev = imix_folds(dist)
        E = {r: evaluate(folds, [pred(f, c) for f in folds]) for r, c in COEF.items()}
        rows = {}
        for r, e in E.items():
            rows[r] = dict(unseen=e["unseen"]["mean"], seen=e["seen"]["mean"], macro=e["macro"]["mean"], coef=COEF[r].tolist(),
                           folds_unseen=list(e["unseen"]["folds"]), folds_seen=list(e["seen"]["folds"]))
        for r in rows:
            for ref in ("ctx_c3", "ctxE_c2hs3p", "ctxE_c2h"):
                du = np.array(rows[r]["folds_unseen"]) - np.array(rows[ref]["folds_unseen"])
                ds = np.array(rows[r]["folds_seen"]) - np.array(rows[ref]["folds_seen"])
                rows[r][f"d_{ref}"] = rows[r]["macro"] - rows[ref]["macro"]
                rows[r][f"better_{ref}"] = int((du < -1e-9).sum() + (ds < -1e-9).sum())
        out[dist] = dict(rows=rows, e_composition_maxdev=dev if dist == "W24-60" else None)
        print(f"\n== {dist}" + (f" (E composition check vs Work E file: max |Δ| {dev:.2e})" if dist == "W24-60" else ""))
        for r, v in rows.items():
            print(f"  {r:28s} unseen {v['unseen']:6.2f} seen {v['seen']:6.2f} dual {v['macro']:6.3f} | Δ c3 {v['d_ctx_c3']:+6.3f} "
                  f"({v['better_ctx_c3']:2d}/10)  Δ c2hs3p {v['d_ctxE_c2hs3p']:+6.3f} ({v['better_ctxE_c2hs3p']:2d}/10)  "
                  f"Δ c2h {v['d_ctxE_c2h']:+6.3f}", flush=True)
    for key, a, b_ in (("M = c3/2 + c2hs3p/2", "ctx_c3", "ctxE_c2hs3p"), ("M' = c3/2 + c2h/2", "ctx_c3", "ctxE_c2h"),
                       ("J = (c3 + c2 + c2hs3p)/3", "ctx_c3", "ctxE_c2hs3p")):
        worst = max(max(out[x]["rows"][key][f"d_{a}"], out[x]["rows"][key][f"d_{b_}"]) for x in DISTS)
        extra = (f", Δ ctx_c2 max {max(out[x]['rows'][key]['macro'] - out[x]['rows']['ctx_c2']['macro'] for x in DISTS):+.3f}"
                 if key.startswith("J") else "")
        print(f"sanity {key}: worst Δ vs its components over {DISTS} = {worst:+.3f} -> {'PASS' if worst <= 0.1 else 'FAIL'} (rule ≤ +0.10){extra}")
    path = OUT / "shiftcv_i_mix.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=2, default=float))


def countdown(x, starts, lens, mode, a, hz):
    """20 Nov research probe: countdown filter on fine (601-cell) remaining-time distributions, per stream, causal. The previous window's
    belief shifted by one 5-s step (cells < 5 s dropped: the stream would have ended; a fraction hz of the >= 600 point mass moves into
    [595, 600) uniformly), renormalised, then combined with the current distribution: 'mix' b = (1 - a) p + a shift(b_prev),
    'prod' b ∝ p * (shift(b_prev) + 1e-6)^a. a: scalar or per-row array (gate; a = 0 -> the row's own distribution, the
    accumulation restarts there). Vectorised over streams by within-stream step."""
    a = np.broadcast_to(np.asarray(a, np.float64), (len(x),))
    if not a.any():
        return x
    y = np.empty_like(x)
    y[starts] = x[starts]
    for j in range(1, int(lens.max())):
        live = lens > j
        cur, prv = starts[live] + j, starts[live] + j - 1
        b = y[prv]
        p = np.zeros_like(b)
        p[:, :595] = b[:, 5:600]
        p[:, 595:600] = (hz * b[:, 600] / 5)[:, None]
        p[:, 600] = (1 - hz) * b[:, 600]
        tot = p.sum(1, keepdims=True)
        p = np.where(tot > 1e-12, p / np.maximum(tot, 1e-300), x[cur])
        ac = a[cur][:, None]
        z = (1 - ac) * x[cur] + ac * p if mode == "mix" else x[cur] * (p + 1e-6) ** ac
        y[cur] = z / z.sum(1, keepdims=True)
    return y


def part_diag():
    """20 Nov research: error anatomy of M (= mixI recipe) under the shifted dual CV (range contributions, signed bias per range, by
    elapsed-since-stream-start) and countdown-filter probes on the existing out-of-fold distributions (no retraining; a = 0 == M).
    Memory-light: folds keep 25-cell grid arrays, the fine arrays are built per fold and dropped. --probes=gated (default) runs the
    gated product filter (active only where M's own prediction >= theta), --probes=none only the anatomy."""
    from stream_utils import RANGE_NAMES, range_index
    cM = 0.5 * np.asarray(IMIX_PK["ctx_c3"]) + 0.5 * np.asarray(IMIX_PK["ctxE_c2hs3p"])
    onset = rec_onsets(d)
    which_probes = ARG.get("--probes", "gated")
    path = OUT / f"shiftcv_diag{TAG}.json"
    out = json.loads(path.read_text()) if path.exists() else {}

    def run(f, filt=None):
        h, s = [x @ FINE for x in f["hgb_g"]], [x @ FINE for x in f["seq_g"]]
        if filt is not None:
            h, s = filt(f, h, s)
        return imix_pred(dict(f, hgb=h, seq=s), cM)

    for dist in DISTS:
        t0 = time.perf_counter()
        folds, _ = imix_folds(dist, lazy=True)
        base = [run(f) for f in folds]
        E0 = evaluate(folds, base)
        res = dict(macro=E0["macro"]["mean"], contrib=dict(zip(RANGE_NAMES, E0["macro"]["contrib"])))
        y = np.concatenate([f["dte"].target.to_numpy() for f in folds])
        p = np.concatenate(base)
        el = np.concatenate([f["dte"].groupby("timeline", sort=False).cumcount().to_numpy() * 5.0 + 5.0 for f in folds])
        ri = range_index(y)
        res["bias"] = {RANGE_NAMES[r]: dict(mean_err=float((p - y)[ri == r].mean()), mae=float(np.abs(p - y)[ri == r].mean()),
                                            mean_pred=float(p[ri == r].mean()), n=int((ri == r).sum())) for r in range(len(RANGE_NAMES))}
        print(f"\n== {dist}: M dual {res['macro']:.3f}  contrib " + " ".join(f"{k} {v:.2f}" for k, v in res["contrib"].items()), flush=True)
        for k, v in res["bias"].items():
            print(f"   {k:10s} mean pred {v['mean_pred']:7.1f}  mean err {v['mean_err']:+7.2f} s  MAE {v['mae']:7.2f}  n {v['n']}", flush=True)
        for th in (150, 250, 350):
            for r in range(len(RANGE_NAMES)):
                m_ = ri == r
                print(f"   P(M pred >= {th}) in {RANGE_NAMES[r]}: {float((p[m_] >= th).mean()):.3f}", flush=True)
        for f in folds:
            tl = f["dte"].timeline.to_numpy()
            ch = np.flatnonzero(np.r_[True, tl[1:] != tl[:-1]])
            f["starts"], f["lens"] = ch, np.diff(np.r_[ch, len(tl)])
            dtr = d[f["tr"]]
            rem = onset.loc[dtr.timeline.to_numpy()].to_numpy() - dtr.stop.to_numpy()
            f["hz"] = float(((rem >= 600) & (rem < 605)).sum() / max((rem >= 600).sum(), 1))
        probes = {}
        settings = [] if which_probes == "none" else [(a_, th) for th in (200, 250, 300, 350) for a_ in (0.2, 0.4, 0.7)]
        for a_, th in settings:
            preds = []
            for f, pb in zip(folds, base):
                gate = a_ * (pb >= th)
                preds.append(run(f, lambda f_, h, s, g=gate: ([countdown(x, f_["starts"], f_["lens"], "prod", g, f_["hz"]) for x in h],
                                                              [countdown(x, f_["starts"], f_["lens"], "prod", g, f_["hz"]) for x in s])))
            E = evaluate(folds, preds)
            du = np.array(E["unseen"]["folds"]) - np.array(E0["unseen"]["folds"])
            ds = np.array(E["seen"]["folds"]) - np.array(E0["seen"]["folds"])
            key = f"prod a={a_} gate M>={th}"
            probes[key] = dict(macro=E["macro"]["mean"], d=E["macro"]["mean"] - res["macro"], better=int((du < 0).sum() + (ds < 0).sum()),
                               folds_unseen=list(E["unseen"]["folds"]), folds_seen=list(E["seen"]["folds"]),
                               contrib=dict(zip(RANGE_NAMES, E["macro"]["contrib"])))
            print(f"   probe {key:24s} dual {E['macro']['mean']:.3f}  Δ {probes[key]['d']:+.3f} ({probes[key]['better']}/10)  contrib "
                  + " ".join(f"{v:.2f}" for v in E["macro"]["contrib"]), flush=True)
        res["folds_unseen"], res["folds_seen"] = list(E0["unseen"]["folds"]), list(E0["seen"]["folds"])
        res["probes"], res["time_s"] = probes, time.perf_counter() - t0
        out[dist] = res
        path.write_text(json.dumps(out, indent=2, default=float))
        del folds, base


def part_e2e():
    """Work L (20 Nov): end-to-end raw-EEG CNN -> GRU (scripts/e2e.py), fold models retrained per seed on the fold's training rows; at
    every checkpoint epoch the held-out recordings are streamed under each distribution (the same shift samples as every other part) ->
    OUT/shiftcv_e2e_<proto>_f<k>.npz, keys <dist>_e<epoch> = seed-mean probabilities (11 classes). Resumable per fold. --seeds=3."""
    from e2e import CFG, predict_e2e, train_e2e
    d2, recs2, F, gain = load_inputs()
    assert (d2.timeline.to_numpy() == d.timeline.to_numpy()).all() and (d2.stop.to_numpy() == d.stop.to_numpy()).all()
    from pathlib import Path
    mask = np.load(Path.home() / "neuralbench" / "save" / "features_base.npz")["split"] != "test"
    X = torch.as_tensor(np.load(Path.home() / "neuralbench" / "save" / "raw_windows.npy", mmap_mode="r")[mask], device=DEV)
    assert len(X) == len(d2)
    n_seeds = int(ARG.get("--seeds", "3"))
    only = ARG.get("--only")  # e.g. unseen:0 (smoke test)
    cfg = dict(CFG, **({"epochs": int(ARG["--epochs"]), "checkpoints": (int(ARG["--epochs"]),)} if "--epochs" in ARG else {}))
    for proto, splits in protocols.items():
        for k, (tr, te) in enumerate(splits):
            if only and only != f"{proto}:{k}":
                continue
            path = OUT / f"shiftcv{TAG}_e2e_{proto}_f{k}.npz"
            if path.exists():
                print(f"e2e {proto} fold {k}: cached", flush=True)
                continue
            t0 = time.perf_counter()
            items = {dist: shifted(proto, k, dist) for dist in DISTS}
            acc = {}

            def on_ck(ep, net, mu, sd):
                for dist, its in items.items():
                    P = predict_e2e(net, mu, sd, F, X, [recs2[i][kk:] for i, kk, _ in its])
                    acc.setdefault(f"{dist}_e{ep}", []).append(P)

            for s_ in range(n_seeds):
                ts = time.perf_counter()
                train_e2e(d2, recs2, F, gain, X, tr, s_, cfg, on_checkpoint=on_ck)
                print(f"e2e {proto} fold {k} seed {s_}: {time.perf_counter() - ts:.0f} s", flush=True)
            np.savez(path, **{key: np.mean(v, axis=0).astype(np.float32) for key, v in acc.items()})
            print(f"e2e {proto} fold {k}: {n_seeds} seeds, {len(DISTS)} distributions in {time.perf_counter() - t0:.0f} s", flush=True)


def part_e2emix():
    """Work L (20 Nov): M (= mixI recipe) vs pre-specified rows with the end-to-end model E2E, per distribution:
    R1 = M with GRU-emb replaced by E2E (same coefficient), R2 = R1 with E2E at twice that coefficient taken from GRU ctx
    (sequence part 0.2 GRU ctx + 0.0667 TCN + 0.1333 E2E); E2E alone (sequence-only decision) for information. The E2E checkpoint epoch
    is chosen per fold by nested LOFO (the epoch minimising the mean of the other 9 folds' R1 score). Acceptance (user, 20 Nov): under D =
    W24-60, Δ vs M <= -0.40 and better in >= 8/10 folds."""
    from decision_layer import slot_median, to_slots
    from stream_utils import RANGE_NAMES
    cM = 0.5 * np.asarray(IMIX_PK["ctx_c3"]) + 0.5 * np.asarray(IMIX_PK["ctxE_c2hs3p"])
    g = cM[5]
    rows_c = {"R1": (cM, g), "R2": (cM - np.array([0, 0, 0, g, 0, g]), 2 * g)}  # (coefs with the GRU-emb slot zeroed later, E2E coef)
    out = {}
    for dist in DISTS:
        folds, _ = imix_folds(dist, lazy=True)
        Z = [np.load(OUT / f"shiftcv{TAG}_e2e_{f['proto']}_f{f['k']}.npz") for f in folds]
        eps = sorted({int(key.split("_e")[1]) for key in Z[0].files if key.startswith(dist + "_e")})

        def pred(f, c, z):
            h, s = [x @ FINE for x in f["hgb_g"]], [x @ FINE for x in f["seq_g"]]
            ws, wh = f["w_old"]
            M = 0.0
            for ci, x in zip(c[:3], h):
                if ci:
                    hh = to_slots(x, f["t"], 0.0) * wh
                    M = M + ci * hh / hh.sum(1, keepdims=True)
            for ci, x in list(zip(c[3:], s)) + ([] if z is None else [z]):
                if ci:
                    q = to_slots(x, f["t"], 0.0) * ws
                    M = M + ci * q / q.sum(1, keepdims=True)
            return slot_median(M)

        base = [pred(f, cM, None) for f in folds]
        E0 = evaluate(folds, base)
        res = {"M": dict(macro=E0["macro"]["mean"], contrib=E0["macro"]["contrib"])}
        ref_u, ref_s = np.array(E0["unseen"]["folds"]), np.array(E0["seen"]["folds"])
        per_ep = {}
        for ep in eps:
            zf = [to_grid(z[f"{dist}_e{ep}"].astype(np.float64), B) @ FINE for z in Z]
            per_ep[ep] = {}
            for rname, (c, ce2e) in rows_c.items():
                c2 = c.copy()
                c2[5] = 0.0
                per_ep[ep][rname] = [pred(f, c2, (ce2e, x)) for f, x in zip(folds, zf)]
            per_ep[ep]["E2E alone"] = [pred(f, np.zeros(6), (1.0, x)) for f, x in zip(folds, zf)]
        for rname in ("R1", "R2", "E2E alone"):
            fold_scores = {ep: [] for ep in eps}
            for ep in eps:
                E = evaluate(folds, per_ep[ep][rname])
                fold_scores[ep] = np.array(E["unseen"]["folds"] + E["seen"]["folds"])
            for ep in eps:
                print(f"  {dist} {rname:9s} epoch {ep:2d}: dual {fold_scores[ep].reshape(2, 5).mean(1).mean():.3f}", flush=True)
            # nested LOFO epoch per fold (selection on R1 for the rows, on itself for E2E alone)
            sel_src = "R1" if rname != "E2E alone" else rname
            if sel_src != rname:
                src_scores = {ep: np.array(evaluate(folds, per_ep[ep][sel_src])["unseen"]["folds"]
                                           + evaluate(folds, per_ep[ep][sel_src])["seen"]["folds"]) for ep in eps}
            else:
                src_scores = fold_scores
            chosen = [min(eps, key=lambda ep: np.delete(src_scores[ep], i).mean()) for i in range(10)]
            assert [f["proto"] for f in folds] == ["unseen"] * 5 + ["seen"] * 5  # fold order == fold_scores order
            preds = [per_ep[chosen[fi]][rname][fi] for fi in range(len(folds))]
            E = evaluate(folds, preds)
            du, ds = np.array(E["unseen"]["folds"]) - ref_u, np.array(E["seen"]["folds"]) - ref_s
            res[rname] = dict(macro=E["macro"]["mean"], d=E["macro"]["mean"] - res["M"]["macro"], better=int((du < 0).sum() + (ds < 0).sum()),
                              epochs=chosen, contrib=E["macro"]["contrib"])
            print(f"== {dist} {rname:9s} nested-epoch dual {E['macro']['mean']:.3f}  Δ vs M {res[rname]['d']:+.3f} ({res[rname]['better']}/10)  "
                  f"epochs {chosen}  contrib " + " ".join(f"{v:.2f}" for v in E["macro"]["contrib"]), flush=True)
        if dist == "W24-60":
            best = min(("R1", "R2"), key=lambda r: res[r]["macro"])
            ok = res[best]["d"] <= -0.40 and res[best]["better"] >= 8
            print(f"ACCEPTANCE (D, best row {best}): Δ {res[best]['d']:+.3f}, {res[best]['better']}/10 -> {'PASS' if ok else 'FAIL'} "
                  f"(rule Δ <= -0.40 and >= 8/10)", flush=True)
        out[dist] = res
    (OUT / f"shiftcv{TAG}_e2emix.json").write_text(json.dumps(out, indent=2, default=float))


def part_seedeval():
    """Work G (19 Nov): HGB-only decision of the member subset, single seed (the first) vs the mean of the seeds, per distribution."""
    from decision_layer import slot_median, to_slots
    onset = rec_onsets(d)
    sfx = f"_{ARG['--members'].replace(',', '+')}_s{ARG['--seeds'].replace(',', '')}"
    seeds = [int(x) for x in ARG["--seeds"].split(",")]
    H = {p: np.load(OUT / f"shiftcv{TAG}_hgbctx{ARG.get('--crop', '')}{sfx}_{p}.npz") for p in protocols}

    def pred_h(f, P, chunk=4096):
        out = np.empty(len(f["t"]))
        _, wh = f["w_old"]
        for i0 in range(0, len(out), chunk):
            sl = slice(i0, i0 + chunk)
            h = to_slots(P[sl], f["t"][sl], 0.0) * wh
            out[sl] = slot_median(h / h.sum(1, keepdims=True))
        return out

    out = {}
    for dist in DISTS:
        folds, cands = [], {f"seed {s}": [] for s in seeds}
        cands["mean"] = []
        for proto, splits in protocols.items():
            for k, (tr, _) in enumerate(splits):
                items = shifted(proto, k, dist)
                idx = np.concatenate([recs[i][kk:] for i, kk, _ in items])
                dte = d.iloc[idx].reset_index(drop=True).copy()
                dte["timeline"] = np.concatenate([[f"{d.timeline[recs[i][0]]}#{j}"] * (len(recs[i]) - kk) for i, kk, j in items])
                dtr = d[tr].reset_index(drop=True)
                _, ob = class_stats(dtr, B)
                f = dict(proto=proto, k=k, dte=dte, t=dte.stop.to_numpy().astype(np.float64),
                         w_old=old_slot_weights(ob, grid_omega(dtr)), D=onset.loc[d.timeline.to_numpy()[idx]].to_numpy())
                folds.append(f)
                Ps = [H[proto][f"{dist}_f{k}_s{s}"].astype(np.float64) @ FINE for s in seeds]
                for s, P in zip(seeds, Ps):
                    cands[f"seed {s}"].append(pred_h(f, P))
                cands["mean"].append(pred_h(f, np.mean(Ps, axis=0)))
        E = {key: evaluate(folds, pr) for key, pr in cands.items()}
        r0 = E[f"seed {seeds[0]}"]
        rows = {}
        for key, e in E.items():
            du = np.array(e["unseen"]["folds"]) - np.array(r0["unseen"]["folds"])
            ds = np.array(e["seen"]["folds"]) - np.array(r0["seen"]["folds"])
            rows[key] = dict(unseen=e["unseen"]["mean"], seen=e["seen"]["mean"], macro=e["macro"]["mean"],
                             delta=e["macro"]["mean"] - r0["macro"]["mean"], better=int((du < -1e-9).sum() + (ds < -1e-9).sum()),
                             d_unseen=du.tolist(), d_seen=ds.tolist())
        out[dist] = rows
        print(f"\n== {dist} (HGB only, members {ARG['--members']}; Δ vs seed {seeds[0]})")
        for key, v in rows.items():
            print(f"  {key:8s} unseen {v['unseen']:6.2f} seen {v['seen']:6.2f} dual {v['macro']:6.2f} Δ {v['delta']:+6.2f} {v['better']:2d}/10 | "
                  f"folds Δ {np.round(v['d_unseen'], 2).tolist()} {np.round(v['d_seen'], 2).tolist()}", flush=True)
    (OUT / f"shiftcv{TAG}_seedeval{sfx}.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    t0 = time.perf_counter()
    {"hgb": lambda: part_hgb(), "seq": lambda: part_seq(POS[1]), "mix": lambda: part_mix(POS[1]), "fmix": lambda: part_fmix(),
     "seedeval": lambda: part_seedeval(), "imix": lambda: part_imix(), "diag": lambda: part_diag(),
     "e2e": lambda: part_e2e(), "e2emix": lambda: part_e2emix()}[PART]()
    print(f"{PART} done ({time.perf_counter() - t0:.0f} s)")
