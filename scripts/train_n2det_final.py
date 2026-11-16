"""Final training of the Work 2 recipe (HGB(c) + CNN N2-detector summaries, sequence models unchanged) on all 540
recordings. Always deterministic (n2det_cv imports repro.set_deterministic). Stages, in order:

  det  (GPU): the CNN N2 detector (n2det_cv.CNNDet: cnn_encoder.Encoder with 2 outputs, seed 0) on all 540
       recordings, epochs = median of the 10 LOFO choices of the dual CV (OUT/n2det_cnn.json); then out-of-fold
       P(N2) for the stream windows of all 540 recordings from 5 subject-grouped inner folds (same epochs), as the
       training rows of n2det_cv.py --inner: the HGB members must never see in-sample detector outputs.
       -> FINAL/det.joblib (state dict + constants), p_oof.npy, p_full.npy (in-sample, checks only), det.json
  hgb  (CPU): ensemble (c) members (ensemble_dist.MEMBERS) on the 817 features + the 5 N2Summary features of p_oof
       (per recording in time order), n_iter per member = median of the dual-CV LOFO choices with n2det
       (OUT/ens_dyn_maxmin_spec_eye_n2det_<member>.npz, rounded to 25), bin edges over all 822 columns.
       -> SAVE/ens_c_all_n2det/model.joblib (kind ens_grid + n2det_* keys)
  mix  --alpha=A [--seq=mix3_all] [--gru-from=DIR]: mix3 bundle = the sequence components of SAVE/<seq>/model.joblib
       (GRU-uw, TCN, CNN encoders, GRU-emb; recipe "seq(old)") + the n2det ensemble, mix_alpha A, n2det_into ();
       --gru-from replaces the GRU-uw states / mu / sd by those of SAVE/<DIR> (e.g. mix3crop_all, crop-augmented GRU)
       -> SAVE/mix3_n2det_all/model.joblib; then build the ZIP with scripts/build_hgb_submission.py.

  detmc: Work 3 multi-class detector, 3 seeds (see stage_detmc).  hgbmc: candidate 2 HGB(c), crop + closeness [--trainval].
  mix also takes --ens=DIR (default ens_c_all_n2det).
Usage: python scripts/train_n2det_final.py det|detmc|hgb|mix [--alpha=0.7] [--seq=mix3_all] [--gru-from=DIR] [--out=mix3_n2det_all]
"""
import json
import sys
import time

import numpy as np

from n2det_cv import CNNDet, sanity  # noqa: F401  (sets deterministic GPU settings on import)

import joblib  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from ablation_c import ABLATIONS, build_table_ext  # noqa: E402
from cv_utils import apply_bin_edges, fit_bin_edges  # noqa: E402
from dist_hgb import decide, to_class  # noqa: E402
from dual_cv import OUT  # noqa: E402
from ensemble_dist import GRID, MEMBERS, grid_omega, to_grid  # noqa: E402
from gru_cv import load_inputs  # noqa: E402
from n2det_stream import N2_NAMES, N2Summary  # noqa: E402
from stream_utils import RANGE_NAMES, summarize  # noqa: E402
from train_hgb import SAVE  # noqa: E402

FINAL = SAVE / "n2det_final"
FINAL.mkdir(parents=True, exist_ok=True)
SCALE, CLIP, EMB, SEED = 20.0, 15.0, 32, 0
FEAT_SET = "dyn_maxmin_spec_eye"


def recordings(d):
    return [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]


def n2_table(p, recs):
    out = np.zeros((len(p), 5), np.float32)
    st = N2Summary(5.0)
    for r in recs:
        st.reset()
        for i in r:
            out[i] = st.update(p[i])
    return out


def stage_det():
    """--trainval (16 Nov, shift calibration): trained on the 500 train+val recordings only -> SAVE/n2det_final_tv/det.joblib,
    p_full (all 540 rows; test rows out-of-sample), p_oof (inner-fold for train+val rows, full model for test rows)."""
    tv = "--trainval" in sys.argv
    final = SAVE / ("n2det_final_tv" if tv else "n2det_final")
    final.mkdir(parents=True, exist_ok=True)
    d, recs, _, _ = load_inputs(include_test=True)
    zp = np.load(SAVE / "n2det_post.npz")
    post = dict(tl=zp["timeline"], k=zp["k"], F=None)
    assert set(post["tl"]) == set(d.timeline) and len(d) == 80975
    cv = json.loads((OUT / "n2det_cnn.json").read_text())
    epochs = int(np.median([f["epoch"] for pr in ("unseen", "seen") for f in cv["folds"][pr]]))
    det = CNNDet(d, post, np.ones(len(d), bool), np.ones(len(zp["k"]), bool))
    det.seed, det.epoch = SEED, epochs
    all_rows = np.arange(len(d))
    trm = (d.split != "test").to_numpy() if tv else np.ones(len(d), bool)
    print(f"{'train+val' if tv else '540'} recordings, {len(d)} stream + {len(zp['k'])} post-onset windows; detector epochs {epochs}, seed {SEED}", flush=True)
    t0 = time.perf_counter()
    p_full, info = det.fit_predict(trm, all_rows)
    state = {k: v.detach().cpu().numpy() for k, v in det.last_model.state_dict().items()}
    print(f"full detector: {info} ({time.perf_counter() - t0:.0f} s)", flush=True)
    p_oof = np.full(len(d), np.nan, np.float32)
    p_oof[~trm] = p_full[~trm]
    rec_tl = d.timeline.to_numpy()
    rows = np.flatnonzero(trm)
    for k, (itr, ite) in enumerate(GroupKFold(5).split(rows, groups=d.subject.to_numpy()[rows])):
        t0 = time.perf_counter()
        mtr = np.zeros(len(d), bool)
        mtr[rows[itr]] = True
        ite = rows[ite]
        p_oof[ite] = det.fit_predict(mtr, ite)[0]
        print(f"  inner fold {k}: {len(np.unique(rec_tl[ite]))} held-out recordings ({time.perf_counter() - t0:.0f} s)", flush=True)
    assert not np.isnan(p_oof).any()
    joblib.dump(dict(state=state, scale=SCALE, clip=CLIP, emb=EMB, epochs=epochs, seed=SEED), final / "det.joblib")
    np.save(final / "p_oof.npy", p_oof)
    np.save(final / "p_full.npy", p_full.astype(np.float32))
    rem = d.target.to_numpy().astype(np.float64)
    res = dict(epochs=epochs, seed=SEED, deterministic=True, n_windows=len(d), oof_sanity=sanity(p_oof.astype(np.float64), rem),
               full_insample_sanity=sanity(p_full.astype(np.float64), rem))
    (final / "det.json").write_text(json.dumps(res, indent=2, default=float))
    s = res["oof_sanity"]
    print(f"OOF (540, subject-grouped): Spearman {s['spearman']:+.3f}, AUC60 {s['auc_last60_vs_300plus']:.3f}, "
          f"AUC30 {s['auc_last30_vs_60_150']:.3f}; in-sample AUC60 {res['full_insample_sanity']['auc_last60_vs_300plus']:.3f}")


def stage_hgb():
    """--trainval: train+val rows only, detector from n2det_final_tv -> SAVE/ens_c_tv_n2det (test check out-of-sample)."""
    tv = "--trainval" in sys.argv
    final = SAVE / ("n2det_final_tv" if tv else "n2det_final")
    wg, sg, _ = ABLATIONS[FEAT_SET]
    df, X, names = build_table_ext(wg, sg)
    recs = recordings(df)
    p_oof, p_full = np.load(final / "p_oof.npy"), np.load(final / "p_full.npy")
    det = joblib.load(final / "det.joblib")
    X = np.concatenate([X, n2_table(p_oof, recs)], axis=1)
    names = list(names) + [f"n2cnn_{s}" for s in N2_NAMES]
    trm = (df.split != "test").to_numpy() if tv else np.ones(len(df), bool)
    bins = fit_bin_edges(X[trm])
    Xb = apply_bin_edges(X[trm], bins)
    y = df.target.to_numpy()[trm]
    models, info = [], {}
    t_all = time.perf_counter()
    for name, (edges, params, _) in MEMBERS.items():
        it = np.load(OUT / f"ens_{FEAT_SET}_n2det_{name}.npz", allow_pickle=True)["n_iter"].item()
        n_iter = int(25 * round(np.median(list(it["unseen"]) + list(it["seen"])) / 25))
        t0 = time.perf_counter()
        clf = HistGradientBoostingClassifier(max_iter=n_iter, max_features=0.5, early_stopping=False, random_state=0, **params)
        clf.fit(Xb, to_class(y, edges))
        info[name] = dict(n_iter=n_iter, lofo=it, fit_s=time.perf_counter() - t0)
        print(f"{name}: {n_iter} iters, fit {info[name]['fit_s']:.0f} s", flush=True)
        models.append((clf, edges))
    omega = grid_omega(df[trm].reset_index(drop=True))
    # in-sample pipeline check on the 40 test recordings, with the full detector's P as the wrapper will produce it
    mt = (df.split == "test").to_numpy()
    d_te = df[mt].reset_index(drop=True)
    Xt = np.concatenate([X[mt, :-5], n2_table(p_full, recs)[mt]], axis=1)
    P = np.mean([to_grid(clf.predict_proba(apply_bin_edges(Xt, bins)), e) for clf, e in models], axis=0)
    score = summarize(d_te, decide(P, None, omega, True, GRID))
    print(f"test (in-sample) hgb_c+n2det {score['wbmae']:.4f}  " + " ".join(f"{r} {score[r]:.1f}" for r in RANGE_NAMES)
          + f"; total fit {time.perf_counter() - t_all:.0f} s, {X.shape[1]} features")
    out = SAVE / ("ens_c_tv_n2det" if tv else "ens_c_all_n2det")
    out.mkdir(exist_ok=True)
    joblib.dump(dict(kind="ens_grid", models=[c for c, _ in models], member_edges=[e for _, e in models], grid=GRID,
                     grid_omega=omega, feature_names=names, bin_edges=bins, calib_x=None, calib_y=None, groups=sg,
                     window_groups=wg, n2det_state=det["state"], n2det_scale=det["scale"], n2det_clip=det["clip"],
                     n2det_emb=det["emb"], n2det_into=()), out / "model.joblib")
    (out / "results.json").write_text(json.dumps(dict(feat_set=FEAT_SET + "+n2det", all540=True, n_features=int(X.shape[1]),
                                                      members=info, detector=dict(epochs=det["epochs"], seed=det["seed"]),
                                                      test_insample=score), indent=2, default=float))
    print("saved", out)


def stage_mix():
    alpha = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--alpha=")), "0.7"))
    seq = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seq=")), "mix3_all")
    ens_dir = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ens=")), "ens_c_all_n2det")
    ens = joblib.load(SAVE / ens_dir / "model.joblib")
    old = joblib.load(SAVE / seq / "model.joblib")
    keys = ("gru_states", "gru_mu", "gru_sd", "gru_cfg", "gru_weighted", "gru_omega_b", "gru_edges", "tcn_states", "tcn_mu",
            "tcn_sd", "tcn_dilations", "cnn_states", "cnn_scale", "cnn_clip", "cnn_emb", "gemb_states", "gemb_mu", "gemb_sd",
            "gru_baseline_n", "gru_elapsed_clip")
    bundle = dict(ens)
    bundle.update({k: old[k] for k in keys}, kind="mix3", mix_alpha=alpha, calib_x=None, calib_y=None)
    gru_from = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--gru-from=")), None)
    if gru_from:
        g = joblib.load(SAVE / gru_from / "model.joblib")
        bundle.update(gru_states=g["gru_states"], gru_mu=g["gru_mu"], gru_sd=g["gru_sd"])
        assert np.allclose(g["gru_omega_b"], old["gru_omega_b"])
        if g.get("gru_n2c"):  # GRU trained with the detector summaries as extra inputs (train_gru_n2c.py)
            bundle["n2det_into"] = ("gru",)
    assert np.allclose(bundle["grid_omega"], old["grid_omega"]) and np.allclose(bundle["grid"], old["grid"])
    out = SAVE / next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), "mix3_n2det_all")
    out.mkdir(exist_ok=True)
    joblib.dump(bundle, out / "model.joblib")
    (out / "results.json").write_text(json.dumps(dict(alpha=alpha, sequence_from=seq, gru_from=gru_from, ens=ens_dir,
                                                      n2det_into=list(bundle["n2det_into"])), indent=2))
    print(f"saved {out}: alpha {alpha}, sequence components from {seq}" + (f", GRU from {gru_from}" if gru_from else "")
          + f", HGB(c) from {ens_dir}, n2det_into {tuple(bundle['n2det_into'])}")



MC_SEEDS = (0, 1, 2)


def stage_detmc():
    """Work 3 multi-class detector (n2det_v2 variant "multi") on all 540 recordings, seeds 0, 1, 2. Epoch per seed = median of
    that seed's dual-CV LOFO choices (OUT/n2det_multi_s<seed>.json, unseen + seen). Per seed: full model (state for the
    wrapper) and out-of-fold outputs for all 540 stream rows from 5 subject-grouped inner folds at the same epoch (features of
    the downstream training rows). Closeness = expected class index / 4; the wrapper averages it over the seeds.
    -> SAVE/n2det_final/detmc.joblib (states, epochs), close_oof.npy / close_full.npy (3-seed means), detmc.json
    --trainval: trained on the 500 train+val recordings only (robustness tests on the 40 test recordings); close_full holds the
    full model's outputs for all 540 rows (test rows out-of-sample), close_oof the inner-fold outputs for the train+val rows
    (test rows = full model) -> SAVE/n2det_final_tv/"""
    from n2det_v2 import VARIANTS, Det
    tv = "--trainval" in sys.argv
    final = SAVE / ("n2det_final_tv" if tv else "n2det_final")
    final.mkdir(parents=True, exist_ok=True)
    d, _, _, _ = load_inputs(include_test=True)
    zp = np.load(SAVE / "n2det_post.npz")
    post = dict(tl=zp["timeline"], k=zp["k"], F=None)
    assert set(post["tl"]) == set(d.timeline) and len(d) == 80975
    det = Det(d, post, np.ones(len(d), bool), np.ones(len(zp["k"]), bool), **VARIANTS["multi"])
    allm = np.ones(len(d), bool)
    trm = (d.split != "test").to_numpy() if tv else allm
    groups = d.subject.to_numpy()
    states, epochs, oof, full = [], [], [], []
    t_all = time.perf_counter()
    for seed in MC_SEEDS:
        cv = json.loads((OUT / f"n2det_multi_s{seed}.json").read_text())
        ep = int(np.median(cv["unseen"]["epochs"] + cv["seen"]["epochs"]))
        t0 = time.perf_counter()
        r = det.fit(trm, allm, seed, stop_epoch=ep)[0][-1]
        states.append({k: v.detach().cpu().numpy() for k, v in det.last_model.state_dict().items()})
        full.append(r[1])
        o = np.full(len(d), np.nan)
        o[~trm] = r[1][~trm]
        rows = np.flatnonzero(trm)
        for itr, ite in GroupKFold(5).split(rows, groups=groups[rows]):
            mtr, mte = np.zeros(len(d), bool), np.zeros(len(d), bool)
            mtr[rows[itr]], mte[rows[ite]] = True, True
            o[mte] = det.fit(mtr, mte, seed, stop_epoch=ep)[0][-1][1]
        assert not np.isnan(o).any()
        oof.append(o)
        epochs.append(ep)
        print(f"seed {seed}: epoch {ep}, full + 5 inner folds in {time.perf_counter() - t0:.0f} s", flush=True)
    c_oof, c_full = np.mean(oof, axis=0), np.mean(full, axis=0)
    rem = d.target.to_numpy().astype(np.float64)
    res = dict(trainval=tv, seeds=list(MC_SEEDS), epochs=epochs, n_cls=det.n_cls, first_pos=det.first_pos, deterministic=True,
               oof_sanity=sanity(c_oof, rem), full_insample_sanity=sanity(c_full, rem), time_s=time.perf_counter() - t_all)
    joblib.dump(dict(states=states, epochs=epochs, seeds=list(MC_SEEDS), n_cls=det.n_cls, first_pos=det.first_pos,
                     scale=SCALE, clip=CLIP, emb=EMB, bip=False), final / "detmc.joblib")
    np.save(final / "close_oof.npy", c_oof.astype(np.float32))
    np.save(final / "close_full.npy", c_full.astype(np.float32))
    (final / "detmc.json").write_text(json.dumps(res, indent=2, default=float))
    s = res["oof_sanity"]
    print(f"closeness OOF (540, subject-grouped, 3-seed mean): Spearman {s['spearman']:+.3f}, AUC60 {s['auc_last60_vs_300plus']:.3f}, "
          f"AUC30 {s['auc_last30_vs_60_150']:.3f}; in-sample Spearman {res['full_insample_sanity']['spearman']:+.3f} ({res['time_s']:.0f} s)")



def stage_hgbmc():
    """Candidate 2 HGB(c): 4 members on the 817 features + 5 summaries of the multi-class detector's closeness, training table
    = original rows + one start-cropped copy of every training recording (hgb_crop_precheck.cropped_copies, seed 0; features
    and summaries re-streamed from the crop), as in the dual CV (ens_crop_cv.py c). n_iter per member = median of that CV's
    LOFO choices, rounded to 25. Closeness of the training rows = out-of-fold (inner folds, stage detmc).
    --trainval: the 500 train+val recordings only, detector from n2det_final_tv (robustness tests) -> SAVE/ens_c_tv_n2c_crop;
    default: all 540 -> SAVE/ens_c_all_n2c_crop"""
    import hgb_crop_precheck as pc
    from n2det_feats import summaries
    tv = "--trainval" in sys.argv
    src = SAVE / ("n2det_final_tv" if tv else "n2det_final")
    det = joblib.load(src / "detmc.joblib")
    c_oof, c_full = np.load(src / "close_oof.npy").astype(np.float64), np.load(src / "close_full.npy").astype(np.float64)
    df, X = pc.df, pc.X817
    assert len(df) == len(c_oof) == 80975
    m = (df.split != "test").to_numpy() if tv else np.ones(len(df), bool)
    assert (np.flatnonzero(m) == np.arange(m.sum())).all()  # train(+val) rows first: d-relative indices == global indices
    d = df[m].reset_index(drop=True)
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    S = summaries(c_oof[m], recs)
    c_rows, c_X, ks = pc.cropped_copies(recs, np.random.default_rng(0))
    crop_recs = [r[k:] for r, k in zip(recs, ks)]
    Sc = summaries(c_oof[m], crop_recs)[c_rows]
    Xtr = np.concatenate([np.concatenate([X[m], S], 1), np.concatenate([c_X, Sc], 1)])
    ytr = np.concatenate([d.target.to_numpy(), d.target.to_numpy()[c_rows]])
    names = list(build_table_ext(*ABLATIONS[FEAT_SET][:2])[2]) + [f"n2close_{s}" for s in N2_NAMES]
    bins = fit_bin_edges(Xtr)
    Xb = apply_bin_edges(Xtr, bins)
    models, info = [], {}
    t_all = time.perf_counter()
    for name, (edges, params, _) in MEMBERS.items():
        it = np.load(OUT / f"ens_dyn_maxmin_spec_eye_n2c_crop_{name}.npz", allow_pickle=True)["n_iter"].item()
        n_iter = int(25 * round(np.median(list(it["unseen"]) + list(it["seen"])) / 25))
        t0 = time.perf_counter()
        clf = HistGradientBoostingClassifier(max_iter=n_iter, max_features=0.5, early_stopping=False, random_state=0, **params)
        clf.fit(Xb, to_class(ytr, edges))
        info[name] = dict(n_iter=n_iter, lofo=it, fit_s=time.perf_counter() - t0)
        print(f"{name}: {n_iter} iters, fit {info[name]['fit_s']:.0f} s", flush=True)
        models.append((clf, edges))
    omega = grid_omega(d)  # class weights from the original rows (as the CV: fold training rows)
    mt = (df.split == "test").to_numpy()
    d_te = df[mt].reset_index(drop=True)
    te_recs = [g.sort_values("stop").index.to_numpy() for _, g in d_te.groupby("timeline", sort=False)]
    Xt = np.concatenate([X[mt], summaries(c_full[mt], te_recs)], 1)  # full model's closeness, as the wrapper computes it
    P = np.mean([to_grid(clf.predict_proba(apply_bin_edges(Xt, bins)), e) for clf, e in models], axis=0)
    score = summarize(d_te, decide(P, None, omega, True, GRID))
    print(f"test ({'out-of-sample' if tv else 'in-sample'}) hgb_c+n2c+crop {score['wbmae']:.4f}  " + " ".join(f"{r} {score[r]:.1f}" for r in RANGE_NAMES)
          + f"; total fit {time.perf_counter() - t_all:.0f} s, {Xtr.shape[1]} features, {len(Xtr)} rows ({len(c_rows)} cropped)")
    out = SAVE / ("ens_c_tv_n2c_crop" if tv else "ens_c_all_n2c_crop")
    out.mkdir(exist_ok=True)
    wg, sg, _ = ABLATIONS[FEAT_SET]
    joblib.dump(dict(kind="ens_grid", models=[c for c, _ in models], member_edges=[e for _, e in models], grid=GRID,
                     grid_omega=omega, feature_names=names, bin_edges=bins, calib_x=None, calib_y=None, groups=sg,
                     window_groups=wg, n2det_states=det["states"], n2det_ncls=det["n_cls"], n2det_first_pos=det["first_pos"],
                     n2det_scale=det["scale"], n2det_clip=det["clip"], n2det_emb=det["emb"], n2det_into=()), out / "model.joblib")
    (out / "results.json").write_text(json.dumps(dict(feat_set=FEAT_SET + "+n2close+crop", trainval=tv, n_features=int(Xtr.shape[1]),
                                                      n_rows=int(len(Xtr)), n_cropped=int(len(c_rows)), members=info,
                                                      detector=dict(epochs=det["epochs"], seeds=det["seeds"]),
                                                      test=score), indent=2, default=float))
    print("saved", out)


if __name__ == "__main__":
    {"det": stage_det, "hgb": stage_hgb, "mix": stage_mix, "detmc": stage_detmc, "hgbmc": stage_hgbmc}[sys.argv[1]]()
