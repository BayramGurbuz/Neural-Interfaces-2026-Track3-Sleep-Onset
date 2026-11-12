"""B3 sanity test: a small causal wake / N1 / N2+ stager trained on HMC crops (b3_hmc_fetch.py) and applied to
the Muse train+val recordings (500; the public test is not used).

Signals (Muse <- HMC): AF8-TP9 <- F4-M1, AF7-TP10 <- C3-M2 (no F3-M2 in HMC), AF7-AF8 <- E1-M2 minus E2-M2.
Per 5 s window (step 5 s, from lights off in HMC / recording start in Muse): scale-free spectral features of the
3 signals (features_b2.spectral12 with the 5 log band powers replaced by log band power relative to the 0.5-30 Hz
total; ratios, peak frequencies, SEF95, aperiodic slope, entropy unchanged) = 36; causal stream: current value,
EMA 30 s / 2 min, and their difference to the recording's first-2-min running mean (as B2Stream) = 216.
No absolute amplitude enters. Labels (HMC): stage of the 30 s epoch containing the window end; N2, N3, R -> N2+.
Model: HistGradientBoostingClassifier (small: 7 leaves, 150 trees, lr 0.05, random_state 0), class-balanced weights.

Reports: HMC leave-recordings-out CV (5 folds by recording) balanced accuracy and N2+ AUC; then the model trained
on all HMC crops applied to Muse:
 1. mean P(N2+) per remaining-time bin (time to the first N2 onset);
 2. Spearman(P(N2+), remaining time) over windows, and the same for the best single existing feature
    (the 817 HGB features of train+val; elapsed time reported separately);
 3. AUC for windows with remaining <= 60 s vs >= 300 s (P(N2+) and the best single existing feature).
Usage: python scripts/b3_stager.py   (writes ~/neuralbench/save/b3_stager.json, b3_stager_muse_p.npy)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ablation_c import ABLATIONS, build_table_ext
from features_b2 import B2Stream, spectral12
from train_hgb import SAVE

CROPS = Path.home() / "neuralbench" / "data" / "hmc" / "crops"
WIN, SFREQ = 640, 128.0
CLASSES = {"W": 0, "N1": 1, "N2": 2, "N3": 2, "R": 2}
PARAMS = dict(max_leaf_nodes=7, max_iter=150, learning_rate=0.05, l2_regularization=1.0, random_state=0,
              class_weight="balanced")


def scale_free(S):
    """(B, 3, T) -> (B, 36): spectral12 with band powers made relative to their sum."""
    f = spectral12(S, SFREQ)
    lbp = f[..., :5]
    tot = np.log10(np.sum(10.0 ** lbp, axis=-1, keepdims=True))
    f = np.concatenate([lbp - tot, f[..., 5:]], axis=-1)
    return f.reshape(len(S), -1)


def stream(F):
    """(L, 36) window features of one recording, time order -> (L, 216) causal features."""
    st = B2Stream(5.0, epoch_group=False)
    return np.stack([st.update(F[i], np.zeros(0)) for i in range(len(F))])


def hmc_recording(path):
    z = np.load(path)
    x = z["data"].astype(np.float64)  # F4-M1, C3-M2, C4-M1, E1-M2, E2-M2 (uV, 128 Hz)
    sig = np.stack([x[0], x[1], x[3] - x[4]])
    n = sig.shape[1] // WIN
    S = sig[:, :n * WIN].reshape(3, n, WIN).transpose(1, 0, 2)
    t_end = (np.arange(n) + 1) * 5.0
    y = np.full(n, -1)
    for o, du, s in zip(z["stage_onset"], z["stage_dur"], z["stage"]):
        m = (t_end > o) & (t_end <= o + du)
        y[m] = CLASSES.get(str(s), -1)
    return stream(scale_free(S)), y, float(z["n2_onset_s"]), t_end


def muse_features():
    """Muse train+val: derivations from the raw windows, causal stream per recording (features_base order)."""
    zb = np.load(SAVE / "features_base.npz")
    ch = [str(c) for c in zb["ch_names"]]
    i = {c: ch.index(c) for c in ("TP9", "AF7", "AF8", "TP10")}
    raw = np.load(SAVE / "raw_windows.npy", mmap_mode="r")
    F = np.empty((len(raw), 36))
    for a in range(0, len(raw), 4096):
        x = np.asarray(raw[a:a + 4096], dtype=np.float64)
        S = np.stack([x[:, i["AF8"]] - x[:, i["TP9"]], x[:, i["AF7"]] - x[:, i["TP10"]], x[:, i["AF7"]] - x[:, i["AF8"]]], 1)
        F[a:a + 4096] = scale_free(S)
    out = np.empty((len(raw), 216))
    tl, stop = zb["timeline"], zb["stop"]
    for t in np.unique(tl):
        idx = np.flatnonzero(tl == t)
        idx = idx[np.argsort(stop[idx])]
        out[idx] = stream(F[idx])
    return out


def auc_sym(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


if __name__ == "__main__":
    recs = sorted(CROPS.glob("SN*.npz"))
    data = [hmc_recording(p) for p in recs]
    print(f"HMC: {len(recs)} recordings, {sum(len(d[1]) for d in data)} windows; class counts "
          f"{np.bincount(np.concatenate([d[1][d[1] >= 0] for d in data]), minlength=3).tolist()}", flush=True)
    # leave-recordings-out CV on HMC
    folds = np.arange(len(data)) % 5
    yt, pt = [], []
    for k in range(5):
        tr = [d for d, f in zip(data, folds) if f != k]
        te = [d for d, f in zip(data, folds) if f == k]
        X = np.concatenate([d[0] for d in tr]); y = np.concatenate([d[1] for d in tr])
        clf = HistGradientBoostingClassifier(**PARAMS).fit(X[y >= 0], y[y >= 0])
        Xt = np.concatenate([d[0] for d in te]); ytt = np.concatenate([d[1] for d in te])
        yt.append(ytt[ytt >= 0]); pt.append(clf.predict_proba(Xt[ytt >= 0]))
    yt, pt = np.concatenate(yt), np.concatenate(pt)
    hmc_bacc = balanced_accuracy_score(yt, pt.argmax(1))
    hmc_auc = roc_auc_score(yt == 2, pt[:, 2])
    print(f"HMC 5-fold (by recording): balanced accuracy {hmc_bacc:.3f}, N2+ vs rest AUC {hmc_auc:.3f}", flush=True)
    X = np.concatenate([d[0] for d in data]); y = np.concatenate([d[1] for d in data])
    clf = HistGradientBoostingClassifier(**PARAMS).fit(X[y >= 0], y[y >= 0])

    # Muse train+val
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X817, names = build_table_ext(wg, sg)
    m = (df.split != "test").to_numpy()
    P = clf.predict_proba(muse_features()[m])
    np.save(SAVE / "b3_stager_muse_p.npy", P.astype(np.float32))
    rem = df.target.to_numpy()[m]
    p2 = P[:, 2]
    bins = [0, 30, 60, 120, 180, 300, 450, 600.1]
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (rem >= lo) & (rem < hi)
        curve.append(dict(range=f"[{lo},{hi:g})", n=int(mm.sum()), p_n2=float(p2[mm].mean()), p_n1=float(P[mm, 1].mean()),
                          p_w=float(P[mm, 0].mean())))
    rho = spearmanr(p2, rem).correlation
    X = X817[m]
    rhos = np.array([spearmanr(X[:, j], rem).correlation for j in range(X.shape[1])])
    rhos = np.nan_to_num(rhos)
    el = names.index("elapsed_s")
    eeg = np.array([j for j in range(len(names)) if j != el])
    jb = eeg[np.argmax(np.abs(rhos[eeg]))]
    lab = np.where(rem <= 60, 1, np.where(rem >= 300, 0, -1))
    mm = lab >= 0
    auc_p = roc_auc_score(lab[mm], p2[mm])
    aucs = np.array([auc_sym(lab[mm], X[mm, j]) for j in eeg])
    jb_auc = eeg[np.argmax(aucs)]
    res = dict(hmc=dict(n_recordings=len(recs), balanced_accuracy=hmc_bacc, n2_auc=hmc_auc), curve=curve,
               spearman=dict(stager_p_n2=rho, best_feature=names[jb], best_feature_rho=float(rhos[jb]),
                             elapsed_rho=float(rhos[el])),
               auc_last60_vs_300plus=dict(stager_p_n2=auc_p, best_feature=names[jb_auc], best_feature_auc=float(aucs.max()),
                                          elapsed_auc=auc_sym(lab[mm], X[mm, el]), n_pos=int((lab == 1).sum()),
                                          n_neg=int((lab == 0).sum())))
    (SAVE / "b3_stager.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
