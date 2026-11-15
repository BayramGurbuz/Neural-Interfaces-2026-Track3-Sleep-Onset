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

Usage: python scripts/train_n2det_final.py det|hgb|mix [--alpha=0.7] [--seq=mix3_all] [--gru-from=DIR] [--out=mix3_n2det_all]
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
    d, recs, _, _ = load_inputs(include_test=True)
    zp = np.load(SAVE / "n2det_post.npz")
    post = dict(tl=zp["timeline"], k=zp["k"], F=None)
    assert set(post["tl"]) == set(d.timeline) and len(d) == 80975
    cv = json.loads((OUT / "n2det_cnn.json").read_text())
    epochs = int(np.median([f["epoch"] for pr in ("unseen", "seen") for f in cv["folds"][pr]]))
    det = CNNDet(d, post, np.ones(len(d), bool), np.ones(len(zp["k"]), bool))
    det.seed, det.epoch = SEED, epochs
    all_rows = np.arange(len(d))
    print(f"540 recordings, {len(d)} stream + {len(zp['k'])} post-onset windows; detector epochs {epochs}, seed {SEED}", flush=True)
    t0 = time.perf_counter()
    p_full, info = det.fit_predict(np.ones(len(d), bool), all_rows)
    state = {k: v.detach().cpu().numpy() for k, v in det.last_model.state_dict().items()}
    print(f"full detector: {info} ({time.perf_counter() - t0:.0f} s)", flush=True)
    p_oof = np.full(len(d), np.nan, np.float32)
    rec_tl = d.timeline.to_numpy()
    for k, (itr, ite) in enumerate(GroupKFold(5).split(all_rows, groups=d.subject.to_numpy())):
        t0 = time.perf_counter()
        mtr = np.zeros(len(d), bool)
        mtr[itr] = True
        p_oof[ite] = det.fit_predict(mtr, ite)[0]
        print(f"  inner fold {k}: {len(np.unique(rec_tl[ite]))} held-out recordings ({time.perf_counter() - t0:.0f} s)", flush=True)
    assert not np.isnan(p_oof).any()
    joblib.dump(dict(state=state, scale=SCALE, clip=CLIP, emb=EMB, epochs=epochs, seed=SEED), FINAL / "det.joblib")
    np.save(FINAL / "p_oof.npy", p_oof)
    np.save(FINAL / "p_full.npy", p_full.astype(np.float32))
    rem = d.target.to_numpy().astype(np.float64)
    res = dict(epochs=epochs, seed=SEED, deterministic=True, n_windows=len(d), oof_sanity=sanity(p_oof.astype(np.float64), rem),
               full_insample_sanity=sanity(p_full.astype(np.float64), rem))
    (FINAL / "det.json").write_text(json.dumps(res, indent=2, default=float))
    s = res["oof_sanity"]
    print(f"OOF (540, subject-grouped): Spearman {s['spearman']:+.3f}, AUC60 {s['auc_last60_vs_300plus']:.3f}, "
          f"AUC30 {s['auc_last30_vs_60_150']:.3f}; in-sample AUC60 {res['full_insample_sanity']['auc_last60_vs_300plus']:.3f}")


def stage_hgb():
    wg, sg, _ = ABLATIONS[FEAT_SET]
    df, X, names = build_table_ext(wg, sg)
    recs = recordings(df)
    p_oof, p_full = np.load(FINAL / "p_oof.npy"), np.load(FINAL / "p_full.npy")
    det = joblib.load(FINAL / "det.joblib")
    X = np.concatenate([X, n2_table(p_oof, recs)], axis=1)
    names = list(names) + [f"n2cnn_{s}" for s in N2_NAMES]
    bins = fit_bin_edges(X)
    Xb = apply_bin_edges(X, bins)
    y = df.target.to_numpy()
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
    omega = grid_omega(df)
    # in-sample pipeline check on the 40 test recordings, with the full detector's P as the wrapper will produce it
    mt = (df.split == "test").to_numpy()
    d_te = df[mt].reset_index(drop=True)
    Xt = np.concatenate([X[mt, :-5], n2_table(p_full, recs)[mt]], axis=1)
    P = np.mean([to_grid(clf.predict_proba(apply_bin_edges(Xt, bins)), e) for clf, e in models], axis=0)
    score = summarize(d_te, decide(P, None, omega, True, GRID))
    print(f"test (in-sample) hgb_c+n2det {score['wbmae']:.4f}  " + " ".join(f"{r} {score[r]:.1f}" for r in RANGE_NAMES)
          + f"; total fit {time.perf_counter() - t_all:.0f} s, {X.shape[1]} features")
    out = SAVE / "ens_c_all_n2det"
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
    ens = joblib.load(SAVE / "ens_c_all_n2det" / "model.joblib")
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
    assert np.allclose(bundle["grid_omega"], old["grid_omega"]) and np.allclose(bundle["grid"], old["grid"])
    out = SAVE / next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), "mix3_n2det_all")
    out.mkdir(exist_ok=True)
    joblib.dump(bundle, out / "model.joblib")
    (out / "results.json").write_text(json.dumps(dict(alpha=alpha, sequence_from=seq, gru_from=gru_from, ens="ens_c_all_n2det",
                                                      n2det_into=list(bundle["n2det_into"])), indent=2))
    print(f"saved {out}: alpha {alpha}, sequence components from {seq}" + (f", GRU from {gru_from}" if gru_from else "")
          + ", HGB(c)+n2det from ens_c_all_n2det")


if __name__ == "__main__":
    {"det": stage_det, "hgb": stage_hgb, "mix": stage_mix}[sys.argv[1]]()
