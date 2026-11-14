"""Work 2: in-domain N2 detector, dual CV, and its sanity test.

Training labels (per training recording):
  positive = the 60 post-onset 5 s windows from the raw files (n2det_extract.py; k < 6 = first 30 s, flagged)
  negative = stream windows with remaining time >= 300 s; windows in between are not used.
Detectors (both window-level, causal: one window in, P(N2) out):
  hgb : HistGradientBoostingClassifier on the 62 window features + 72 bipolar ("a") features
  cnn : cnn_encoder.Encoder (same architecture) with 2 outputs on the raw (4, 640) window, deterministic GPU settings;
        epoch per fold chosen LOFO on the other folds' held-out binary CE (labelled windows only)
Leakage rule: in every outer fold (5 unseen + 5 seen) the detector sees only that fold's training recordings and
produces P(N2) for the held-out recordings' stream windows. With --inner, the training recordings get out-of-fold
P(N2) too (inner 5-fold: grouped by subject in the unseen protocol, by recording in the seen protocol, mimicking
how the held-out recordings relate to the detector's training set) -> features for the downstream models.

Sanity test (unseen-protocol OOF, covers all 500 train+val recordings once; same measures as B3):
  mean P(N2) by remaining time; Spearman(P, remaining); AUC last 60 s vs >= 300 s; AUC last 30 s vs 60-150 s.
  Baseline: best single feature of the 817 (in-sample choice, |rho| / symmetric AUC), and elapsed time.

Usage: python scripts/n2det_cv.py hgb|cnn [--inner]
  -> OUT/n2det_<det>_<proto>_outer.npz (per fold P on held-out stream rows), OUT/n2det_<det>_oof.npy (unseen OOF),
     OUT/n2det_<det>.json; with --inner OUT/n2det_<det>_<proto>_f<k>.npy (P for all train+val stream rows, fold k)
"""
import json
import sys
import time

from repro import set_deterministic

set_deterministic()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from torch import nn  # noqa: E402

from ablation_c import ABLATIONS, build_table_ext  # noqa: E402
from cnn_encoder import BATCH, GAIN_SD, LR, NOISE_SD, WD, Encoder, infer  # noqa: E402
from dual_cv import OUT, seen_splits, unseen_splits  # noqa: E402
from gru_cv import DEV, load_inputs  # noqa: E402
from train_hgb import SAVE  # noqa: E402

PARAMS = dict(max_leaf_nodes=15, max_iter=300, learning_rate=0.05, l2_regularization=1.0, max_features=0.5,
              random_state=0, class_weight="balanced")
CNN_EPOCHS = 20
NEG_MIN = 300.0


def load_all():
    """Stream rows (train+val) and post-onset rows (train+val) with features, raw windows and recording ids."""
    d, recs, F62, _ = load_inputs()
    zb = np.load(SAVE / "features_base.npz")
    m = zb["split"] != "test"
    Fb = np.load(SAVE / "features_b2.npz")["feats_bip"][m].astype(np.float32)
    zp = np.load(SAVE / "n2det_post.npz")
    mp = zp["split"] != "test"
    post = dict(tl=zp["timeline"][mp], k=zp["k"][mp], F=np.concatenate([zp["feats62"][mp], zp["feats_bip"][mp]], 1))
    assert set(post["tl"]) == set(d.timeline)
    return d, np.concatenate([F62, Fb], 1), post, m, mp


def train_rows(d, post, tr):
    """Training set of the detector for the training mask tr over stream rows."""
    rec_tr = np.unique(d.timeline.to_numpy()[tr])
    neg = np.flatnonzero(tr & (d.target.to_numpy() >= NEG_MIN))
    pos = np.flatnonzero(np.isin(post["tl"], rec_tr))
    return neg, pos


class HGBDet:
    def __init__(self, d, Fs, post):
        self.d, self.Fs, self.post = d, Fs, post

    def fit_predict(self, tr, te_rows):
        neg, pos = train_rows(self.d, self.post, tr)
        X = np.concatenate([self.Fs[neg], self.post["F"][pos]])
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        clf = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
        return clf.predict_proba(self.Fs[te_rows])[:, 1], dict(n_neg=len(neg), n_pos=len(pos))


class CNNDet:
    """Encoder(n_out=2); stream raw windows and post-onset raw windows stacked on the GPU."""

    def __init__(self, d, post, m, mp):
        self.d, self.post = d, post
        rs = np.load(SAVE / "raw_windows.npy", mmap_mode="r")[m]
        rp = np.load(SAVE / "n2det_post.npz")["raw"][mp]
        self.ns = len(rs)
        self.X = torch.as_tensor(np.concatenate([rs, rp]), device=DEV)
        self.seed = 0
        self.epoch = None  # fixed epoch (inner folds / final); None = return all epochs

    def _train(self, idx_tr, y_tr, eval_sets):
        torch.manual_seed(self.seed)
        g = torch.Generator(device=DEV).manual_seed(self.seed)
        idx_tr = torch.as_tensor(idx_tr, device=DEV)
        y_tr = torch.as_tensor(y_tr, device=DEV).long()
        model = Encoder(n_out=2).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CNN_EPOCHS)
        ce = nn.CrossEntropyLoss(weight=torch.tensor([len(y_tr) / (2 * (y_tr == 0).sum()),
                                                      len(y_tr) / (2 * (y_tr == 1).sum())], device=DEV).float())
        out = []
        n_ep = CNN_EPOCHS if self.epoch is None else self.epoch
        for ep in range(n_ep):
            model.train()
            perm = torch.randperm(len(idx_tr), generator=g, device=DEV)
            for b in range(0, len(perm), BATCH):
                j = perm[b:b + BATCH]
                x = self.X[idx_tr[j]]
                gain = torch.pow(10.0, GAIN_SD * torch.randn(len(j), 4, 1, generator=g, device=DEV))
                x = torch.clamp(x * gain / 20.0, -15.0, 15.0) + NOISE_SD * torch.randn(x.shape, generator=g, device=DEV)
                loss = ce(model(x)[0], y_tr[j])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            if self.epoch is None or ep == n_ep - 1:
                out.append([torch.softmax(infer(model, self.X, torch.as_tensor(e, device=DEV))[0], 1)[:, 1].cpu().numpy()
                            for e in eval_sets])
        return out

    def fit_predict(self, tr, te_rows, te_mask=None):
        neg, pos = train_rows(self.d, self.post, tr)
        idx = np.r_[neg, self.ns + pos]
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        evals = [te_rows]
        if te_mask is not None:  # labelled held-out windows for the epoch choice
            nte, pte = train_rows(self.d, self.post, te_mask)
            evals.append(np.r_[nte, self.ns + pte])
            yte = np.r_[np.zeros(len(nte)), np.ones(len(pte))]
        res = self._train(idx, y, evals)
        info = dict(n_neg=len(neg), n_pos=len(pos))
        if self.epoch is not None:
            return res[-1][0], info
        info["ce_out"] = [float(-np.mean(yte * np.log(r[1] + 1e-7) + (1 - yte) * np.log(1 - r[1] + 1e-7))) for r in res]
        return [r[0] for r in res], info


def sanity(p, rem):
    bins = [0, 30, 60, 90, 150, 300, 450, 600.1]
    curve = [dict(range=f"[{lo},{hi:g})", n=int(((rem >= lo) & (rem < hi)).sum()),
                  p=float(p[(rem >= lo) & (rem < hi)].mean())) for lo, hi in zip(bins[:-1], bins[1:])]
    m1 = (rem <= 60) | (rem >= 300)
    m2 = (rem <= 30) | ((rem >= 60) & (rem <= 150))
    return dict(curve=curve, spearman=float(spearmanr(p, rem).correlation),
                auc_last60_vs_300plus=float(roc_auc_score(rem[m1] <= 60, p[m1])),
                auc_last30_vs_60_150=float(roc_auc_score(rem[m2] <= 30, p[m2])))


def baseline(d):
    """Best single feature of the 817 (in-sample, symmetric) on the same measures; cached."""
    path = OUT / "n2det_baseline.json"
    if path.exists():
        return json.loads(path.read_text())
    wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
    df, X, names = build_table_ext(wg, sg)
    m = (df.split != "test").to_numpy()
    X, rem = X[m], df.target.to_numpy()[m]
    assert (df[m].timeline.to_numpy() == d.timeline.to_numpy()).all()
    m1 = (rem <= 60) | (rem >= 300)
    m2 = (rem <= 30) | ((rem >= 60) & (rem <= 150))
    out = {}
    for key, f in (("spearman", lambda x: abs(spearmanr(x, rem).correlation)),
                   ("auc_last60_vs_300plus", lambda x: roc_auc_score(rem[m1] <= 60, x[m1])),
                   ("auc_last30_vs_60_150", lambda x: roc_auc_score(rem[m2] <= 30, x[m2]))):
        v = np.nan_to_num(np.array([f(X[:, j]) for j in range(X.shape[1])]), nan=0.5 if "auc" in key else 0.0)
        if "auc" in key:
            v = np.maximum(v, 1 - v)
        el = names.index("elapsed_s")
        eeg = [j for j in range(len(names)) if j != el]
        jb = eeg[int(np.argmax(v[eeg]))]
        out[key] = dict(best_feature=names[jb], best=float(v[jb]), elapsed=float(v[el]))
    path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    det = sys.argv[1]
    inner = "--inner" in sys.argv
    d, Fs, post, m, mp = load_all()
    model = HGBDet(d, Fs, post) if det == "hgb" else CNNDet(d, post, m, mp)
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    rem = d.target.to_numpy().astype(np.float64)
    tl = d.timeline.to_numpy()
    t_all = time.perf_counter()
    res = {"params": PARAMS if det == "hgb" else dict(epochs=CNN_EPOCHS), "folds": {}}
    for proto, splits in protocols.items():
        outs, curves, infos = [], [], []
        for k, (tr, te) in enumerate(splits):
            t0 = time.perf_counter()
            te_rows = np.flatnonzero(te)
            if det == "hgb":
                p, info = model.fit_predict(tr, te_rows)
            else:
                p, info = model.fit_predict(tr, te_rows, te_mask=te)
                curves.append(info["ce_out"])
            outs.append(p)
            infos.append(info)
            print(f"  {proto} fold {k}: {time.perf_counter() - t0:.0f} s {dict((a, b) for a, b in info.items() if a != 'ce_out')}",
                  flush=True)
        if det == "cnn":  # LOFO epoch per fold on the other folds' held-out CE
            C = np.array(curves)
            sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
            outs = [outs[k][j] for k, j in enumerate(sel)]
            for k, j in enumerate(sel):
                infos[k]["epoch"] = j + 1
                infos[k]["ce_out_sel"] = float(C[k, j])
                del infos[k]["ce_out"]
            res.setdefault("ce_curve_mean", {})[proto] = C.mean(0).tolist()
        np.savez(OUT / f"n2det_{det}_{proto}_outer.npz", *outs)
        res["folds"][proto] = infos
        if inner:  # out-of-fold P(N2) for the training recordings (inner folds) + held-out rows from the outer model
            for k, (tr, te) in enumerate(splits):
                t0 = time.perf_counter()
                full = np.full(len(d), np.nan, np.float32)
                full[te] = outs[k]
                rows_tr = np.flatnonzero(tr)
                groups = (d.subject.to_numpy() if proto == "unseen" else tl)[rows_tr]
                if det == "cnn":
                    model.epoch = infos[k]["epoch"]
                for itr, ite in GroupKFold(5).split(rows_tr, groups=groups):
                    mtr = np.zeros(len(d), bool)
                    mtr[rows_tr[itr]] = True
                    full[rows_tr[ite]] = model.fit_predict(mtr, rows_tr[ite])[0]
                model.epoch = None
                assert not np.isnan(full).any()
                np.save(OUT / f"n2det_{det}_{proto}_f{k}.npy", full)
                print(f"  inner {proto} fold {k}: {time.perf_counter() - t0:.0f} s", flush=True)
    oof = np.full(len(d), np.nan)
    for (tr, te), p in zip(protocols["unseen"], np.load(OUT / f"n2det_{det}_unseen_outer.npz").values()):
        oof[te] = p
    assert not np.isnan(oof).any()
    np.save(OUT / f"n2det_{det}_oof.npy", oof.astype(np.float32))
    s = sanity(oof, rem)
    seen_te = np.concatenate([np.flatnonzero(te) for _, te in protocols["seen"]])
    seen_p = np.concatenate(list(np.load(OUT / f"n2det_{det}_seen_outer.npz").values()))
    s_seen = sanity(seen_p, rem[seen_te])
    zp = np.load(SAVE / "n2det_post.npz")
    res.update(sanity_unseen=s, sanity_seen=dict((a, b) for a, b in s_seen.items() if a != "curve"),
               baseline=baseline(d), time_s=time.perf_counter() - t_all)
    path = OUT / f"n2det_{det}.json"
    path.write_text(json.dumps(res, indent=2, default=float))
    b = res["baseline"]
    print(f"\n{det}: unseen OOF  Spearman {s['spearman']:+.3f} (best feature |rho| {b['spearman']['best']:.3f}, elapsed "
          f"{b['spearman']['elapsed']:.3f})")
    print(f"  AUC last 60 vs >=300: {s['auc_last60_vs_300plus']:.3f} (best {b['auc_last60_vs_300plus']['best']:.3f}, "
          f"elapsed {b['auc_last60_vs_300plus']['elapsed']:.3f})")
    print(f"  AUC last 30 vs 60-150: {s['auc_last30_vs_60_150']:.3f} (best {b['auc_last30_vs_60_150']['best']:.3f}, "
          f"elapsed {b['auc_last30_vs_60_150']['elapsed']:.3f})")
    print(f"  seen folds: Spearman {s_seen['spearman']:+.3f}, AUC60 {s_seen['auc_last60_vs_300plus']:.3f}, "
          f"AUC30 {s_seen['auc_last30_vs_60_150']:.3f}")
    print("  curve " + " ".join(f"{c['range']} {c['p']:.3f}" for c in s["curve"]))
    print(f"  best features: {b['spearman']['best_feature']}, {b['auc_last60_vs_300plus']['best_feature']}, "
          f"{b['auc_last30_vs_60_150']['best_feature']} ({res['time_s']:.0f} s)")
