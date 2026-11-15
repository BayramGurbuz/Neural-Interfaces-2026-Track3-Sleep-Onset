"""Work 3, stage 1: screening of N2-detector variants on the detector sanity measures (cheap, GPU only).

Same protocol as n2det_cv.py cnn (unseen-protocol outer folds: the detector sees only the fold's training recordings,
P(N2) for the held-out recordings' stream windows; deterministic; epoch per fold LOFO on the other folds' held-out binary
CE over the labelled windows). Variants (each changes one thing against the shipped detector = "base"):
  base     Encoder(2) on the 4 raw channels; negative = remaining >= 300 s, positive = the 60 post-onset windows
  seeds3   mean P of base trained with seeds 0, 1, 2 (each with its own LOFO epoch)
  bip      + 4 bipolar derivations as input channels (AF7-AF8, AF8-TP10, AF7-TP9, TP9-TP10; by channel name) -> 8 channels
  first30  3 classes: negative / post-onset first 30 s (k < 6) / post-onset later; P(N2) = 1 - P(negative)
  multi    5 classes: remaining >= 300 / [150, 300) / [60, 150) / [0, 60) / post-onset (the in-between stream windows are
           used with their remaining-time class); P(N2) = P(post-onset); second score "close" = expected class index / 4
  all      multi + first30 + bip (6 classes, 8 channels)
Class-balanced CE in all variants (as base). Measures (unseen OOF, 500 train+val recordings): Spearman(P, remaining),
AUC last 60 s vs >= 300 s, AUC last 30 s vs 60-150 s, mean P by remaining time. The winner goes to the B-member
ablation with inner-fold P (n2det_cv.py --inner + n2det_ablation.py); nothing here touches a submission.

Usage: python scripts/n2det_v2.py [variant ...]   -> OUT/n2det_v2.json (resumable per variant), OUT/n2det_v2_<variant>_oof.npy
"""
import json
import sys
import time

import numpy as np

from n2det_cv import CNN_EPOCHS, NEG_MIN, load_all, sanity, train_rows  # noqa: F401  (sets deterministic GPU settings)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from cnn_encoder import BATCH, GAIN_SD, LR, NOISE_SD, WD, Encoder  # noqa: E402
from dual_cv import OUT, unseen_splits  # noqa: E402
from gru_cv import DEV  # noqa: E402
from train_hgb import SAVE  # noqa: E402

VARIANTS = {"base": dict(), "bip": dict(bip=True), "first30": dict(first30=True), "multi": dict(multi=True),
            "all": dict(bip=True, first30=True, multi=True)}
MID_EDGES = (150.0, 60.0)  # remaining-time classes of the in-between windows: [150, 300) -> 1, [60, 150) -> 2, [0, 60) -> 3
BIP = (("AF7", "AF8"), ("AF8", "TP10"), ("AF7", "TP9"), ("TP9", "TP10"))


class Det:
    def __init__(self, d, post, m, mp, bip=False, first30=False, multi=False):
        self.d, self.post, self.bip, self.first30, self.multi = d, post, bip, first30, multi
        rs = np.load(SAVE / "raw_windows.npy", mmap_mode="r")[m]
        rp = np.load(SAVE / "n2det_post.npz")["raw"][mp]
        self.ns = len(rs)
        self.X = torch.as_tensor(np.concatenate([rs, rp]), device=DEV)
        ch = [str(c) for c in np.load(SAVE / "features_base.npz")["ch_names"]]
        self.pairs = [(ch.index(a), ch.index(b)) for a, b in BIP]
        self.rem = d.target.to_numpy().astype(np.float64)

    def labelled(self, mask):
        """Rows (into self.X) and class labels of the recordings whose stream rows are in mask."""
        neg, pos = train_rows(self.d, self.post, mask)
        idx, y = [neg], [np.zeros(len(neg), int)]
        c = 0
        if self.multi:
            mid = np.flatnonzero(mask & (self.rem < NEG_MIN))
            idx.append(mid)
            y.append(1 + (self.rem[mid] < MID_EDGES[0]).astype(int) + (self.rem[mid] < MID_EDGES[1]).astype(int))
            c = 3
        idx.append(self.ns + pos)
        y.append(c + 1 + ((self.post["k"][pos] >= 6).astype(int) if self.first30 else np.zeros(len(pos), int)))
        self.n_cls = c + (3 if self.first30 else 2)
        self.first_pos = c + 1  # classes >= first_pos are post-onset
        return np.concatenate(idx), np.concatenate(y)

    def inputs(self, x):
        if not self.bip:
            return x
        return torch.cat([x] + [x[:, a:a + 1] - x[:, b:b + 1] for a, b in self.pairs], 1)

    @torch.no_grad()
    def probs(self, model, idx, bs=2048):
        model.eval()
        out = [torch.softmax(model(torch.clamp(self.inputs(self.X[idx[b:b + bs]]) / 20.0, -15.0, 15.0))[0], 1)
               for b in range(0, len(idx), bs)]
        return torch.cat(out).cpu().numpy().astype(np.float64)

    def fit(self, tr, te, seed, stop_epoch=None):
        """-> per epoch: (P(N2), closeness) on the held-out stream rows, binary CE on the held-out labelled windows.
        stop_epoch (inner folds / final): same cosine schedule, stopped after that many epochs, evaluated once at the end."""
        idx_tr, y_tr = self.labelled(tr)
        idx_ev, y_ev = self.labelled(te)
        keep = (y_ev == 0) | (y_ev >= self.first_pos)  # binary CE on negative / post-onset windows only (comparable across variants)
        idx_ev, yb = idx_ev[keep], (y_ev[keep] >= self.first_pos).astype(float)
        te_rows = torch.as_tensor(np.flatnonzero(te), device=DEV)
        idx_ev = torch.as_tensor(idx_ev, device=DEV)
        torch.manual_seed(seed)
        g = torch.Generator(device=DEV).manual_seed(seed)
        idx_t = torch.as_tensor(idx_tr, device=DEV)
        y_t = torch.as_tensor(y_tr, device=DEV).long()
        model = Encoder(n_out=self.n_cls)
        if self.bip:
            model.conv[0] = nn.Conv1d(4 + len(self.pairs), 16, 7, stride=2, padding=3)
        model = model.to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CNN_EPOCHS)
        cnt = torch.bincount(y_t, minlength=self.n_cls).float()
        ce = nn.CrossEntropyLoss(weight=len(y_t) / (self.n_cls * cnt))
        n_in = 4 + (len(self.pairs) if self.bip else 0)
        w_close = np.minimum(np.arange(self.n_cls), self.first_pos) / self.first_pos
        out = []
        n_ep = CNN_EPOCHS if stop_epoch is None else stop_epoch
        for ep in range(n_ep):
            model.train()
            perm = torch.randperm(len(idx_t), generator=g, device=DEV)
            for b in range(0, len(perm), BATCH):
                j = perm[b:b + BATCH]
                x = self.inputs(self.X[idx_t[j]])
                gain = torch.pow(10.0, GAIN_SD * torch.randn(len(j), n_in, 1, generator=g, device=DEV))
                x = torch.clamp(x * gain / 20.0, -15.0, 15.0) + NOISE_SD * torch.randn(x.shape, generator=g, device=DEV)
                loss = ce(model(x)[0], y_t[j])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            if stop_epoch is not None and ep < n_ep - 1:
                continue
            P = self.probs(model, te_rows)
            pe = self.probs(model, idx_ev)[:, self.first_pos:].sum(1)
            out.append((P[:, self.first_pos:].sum(1), P @ w_close,
                        float(-np.mean(yb * np.log(pe + 1e-7) + (1 - yb) * np.log(1 - pe + 1e-7)))))
        return out, dict(n_train=len(y_tr), classes=np.bincount(y_tr, minlength=self.n_cls).tolist())


def run_variant(det, d, splits, seed):
    """LOFO epoch -> OOF P(N2), OOF closeness, epochs, CE."""
    runs, infos = [], []
    for k, (tr, te) in enumerate(splits):
        t0 = time.perf_counter()
        r, info = det.fit(tr, te, seed)
        runs.append(r)
        infos.append(info)
        print(f"    fold {k}: {time.perf_counter() - t0:.0f} s {info}", flush=True)
    C = np.array([[e[2] for e in r] for r in runs])
    sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
    p, close = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for k, (_, te) in enumerate(splits):
        p[te], close[te] = runs[k][sel[k]][0], runs[k][sel[k]][1]
    assert not np.isnan(p).any()
    return p, close, [s + 1 for s in sel], [float(C[k, s]) for k, s in enumerate(sel)]


def line(name, s):
    return (f"{name:10s} Spearman {s['spearman']:+.3f}  AUC60 {s['auc_last60_vs_300plus']:.3f}  AUC30 {s['auc_last30_vs_60_150']:.3f}  curve "
            + " ".join(f"{c['p']:.3f}" for c in s["curve"]))


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("--")] or ["base", "seeds3", "bip", "first30", "multi", "all"]
    path = OUT / "n2det_v2.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    d, _, post, m, mp = load_all()
    splits = unseen_splits(d)
    rem = d.target.to_numpy().astype(np.float64)

    def save(name, entry):
        res[name] = entry
        path.write_text(json.dumps(res, indent=2, default=float))
        print(line(name, entry["sanity"]) + (f"\n{'':10s} close: " + line("", entry["sanity_close"]).strip() if "sanity_close" in entry else "")
              + f"  epochs {entry.get('epochs')} ({entry.get('time_s', 0):.0f} s)", flush=True)

    for name in want:
        if name in res:
            print(line(name, res[name]["sanity"]) + "  (cached)", flush=True)
            continue
        t0 = time.perf_counter()
        if name == "seeds3":
            det = Det(d, post, m, mp)
            ps = []
            for seed in (0, 1, 2):
                f = OUT / f"n2det_v2_base_s{seed}_oof.npy"
                if not f.exists():
                    print(f"  base seed {seed}", flush=True)
                    np.save(f, run_variant(det, d, splits, seed)[0].astype(np.float32))
                ps.append(np.load(f).astype(np.float64))
            p = np.mean(ps, axis=0)
            np.save(OUT / "n2det_v2_seeds3_oof.npy", p.astype(np.float32))
            save(name, dict(sanity=sanity(p, rem), single_seeds=[sanity(q, rem) for q in ps], time_s=time.perf_counter() - t0))
            del det
        else:
            print(f"  {name}", flush=True)
            det = Det(d, post, m, mp, **VARIANTS[name])
            p, close, epochs, ce = run_variant(det, d, splits, 0)
            np.save(OUT / f"n2det_v2_{name}{'_s0' if name == 'base' else ''}_oof.npy", p.astype(np.float32))
            entry = dict(sanity=sanity(p, rem), epochs=epochs, ce_out_sel=ce, time_s=time.perf_counter() - t0)
            if VARIANTS[name].get("multi"):
                entry["sanity_close"] = sanity(close, rem)
            save(name, entry)
            del det
        torch.cuda.empty_cache()
    print("\nshipped detector (n2det_cv cnn): Spearman -0.479  AUC60 0.945  AUC30 0.815")
    for name in res:
        print(line(name, res[name]["sanity"]))
