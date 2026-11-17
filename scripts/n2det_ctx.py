"""Work C (16 Nov): multi-class N2 detector with context of the last N windows, screening on the detector sanity measures.

Model: the window CNN (cnn_encoder.Encoder, its 32-d embedding) applied to each of the last N windows (current included); a
1-layer GRU (32) over the context embeddings; output = Linear([current embedding, GRU last state]) -> 5 classes (as n2det_v2
"multi": remaining >= 300 / [150, 300) / [60, 150) / [0, 60) / post-onset). Closeness = expected class index / 4.
Context = only windows the stream has delivered: positions before the stream start are zero windows (left padding), so a
mid-recording start just shortens the context. Training: with probability 0.5 the context is cut to a random length 1..N
(emulates a stream start); post-onset windows continue the recording's timeline (stream windows, then the 60 post-onset ones).
Inference on held-out recordings: context from the recording start (k = 0) and from a start at window 2 (shift check).
Protocol as n2det_v2 (unseen outer folds, LOFO epoch on the other folds' held-out binary CE, class-balanced CE, gain / noise
augmentation, deterministic, seed 0). Reference (single window, seed 0): Spearman -0.521, AUC60 0.951, AUC30 0.821.
Stop rule (user): Spearman must improve by >= 0.03 (<= -0.551).

Usage: python scripts/n2det_ctx.py N [N ...]   -> OUT/n2det_ctx.json (resumable per N), OUT/n2det_ctx<N>_oof.npy
"""
import json
import sys
import time

import numpy as np

from n2det_cv import CNN_EPOCHS, NEG_MIN, load_all, sanity  # noqa: F401  (sets deterministic GPU settings)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from cnn_encoder import BATCH, GAIN_SD, LR, NOISE_SD, WD, Encoder  # noqa: E402
from dual_cv import OUT, unseen_splits  # noqa: E402
from gru_cv import DEV  # noqa: E402
from train_hgb import SAVE  # noqa: E402

MID_EDGES = (150.0, 60.0)
N_CLS, FIRST_POS = 5, 4


class CtxDet(nn.Module):
    def __init__(self, n_ctx):
        super().__init__()
        self.n_ctx = n_ctx
        self.enc = Encoder(n_out=N_CLS)
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.out = nn.Linear(64, N_CLS)

    def forward(self, x, valid):
        """x: (B, N, 4, 640) scaled windows (zeros where invalid), valid: (B, N) bool (left-padded) -> (B, N_CLS) logits."""
        Bn, N = x.shape[:2]
        e = self.enc(x.reshape(Bn * N, *x.shape[2:]))[1].reshape(Bn, N, -1) * valid[..., None]
        h = self.gru(e)[1][-1]
        return self.out(torch.cat([e[:, -1], h], 1))


class Data:
    def __init__(self, d, post, m, mp):
        rs = np.load(SAVE / "raw_windows.npy", mmap_mode="r")[m]
        zp = np.load(SAVE / "n2det_post.npz")
        rp = zp["raw"][mp]
        self.ns = len(rs)
        self.X = torch.as_tensor(np.concatenate([rs, rp]), device=DEV)
        rem = d.target.to_numpy().astype(np.float64)
        y_stream = np.where(rem >= NEG_MIN, 0, 1 + (rem < MID_EDGES[0]).astype(int) + (rem < MID_EDGES[1]).astype(int))
        ptl, pk = post["tl"], post["k"]
        self.seq, self.lab, self.n_stream = {}, {}, {}
        for tl, g in d.groupby("timeline", sort=False):
            s_rows = g.sort_values("stop").index.to_numpy()
            p_rows = np.flatnonzero(ptl == tl)
            p_rows = p_rows[np.argsort(pk[p_rows])]
            self.seq[tl] = np.concatenate([s_rows, self.ns + p_rows])
            self.lab[tl] = np.concatenate([y_stream[s_rows], np.full(len(p_rows), N_CLS - 1)])
            self.n_stream[tl] = len(s_rows)
        self.tl_of_row = d.timeline.to_numpy()

    def items(self, tls):
        """All labelled (recording, position) pairs of the recordings tls -> arrays tl index, pos, label."""
        T, P, Y = [], [], []
        for i, tl in enumerate(tls):
            n = len(self.seq[tl])
            T.append(np.full(n, i)); P.append(np.arange(n)); Y.append(self.lab[tl])
        return np.concatenate(T), np.concatenate(P), np.concatenate(Y)

    def context(self, tls, ti, pos, n_ctx, start=None):
        """Index (B, N) into X (-1 = padding) and validity mask; start: per item first allowed position (stream start)."""
        idx = np.full((len(ti), n_ctx), -1, np.int64)
        for b, (t, p) in enumerate(zip(ti, pos)):
            s = self.seq[tls[t]]
            lo = min(max(p - n_ctx + 1, 0 if start is None else start[b]), p)  # a position before the start: itself only
            idx[b, n_ctx - (p - lo + 1):] = s[lo:p + 1]
        return idx, idx >= 0

    def batch(self, idx, valid, gen=None, augment=False):
        x = self.X[torch.as_tensor(np.maximum(idx, 0), device=DEV)]  # (B, N, 4, 640)
        if augment:
            gain = torch.pow(10.0, GAIN_SD * torch.randn(x.shape[0], 1, 4, 1, generator=gen, device=DEV))
            x = torch.clamp(x * gain / 20.0, -15.0, 15.0) + NOISE_SD * torch.randn(x.shape, generator=gen, device=DEV)
        else:
            x = torch.clamp(x / 20.0, -15.0, 15.0)
        v = torch.as_tensor(valid, device=DEV)
        return x * v[..., None, None], v.float()


def run_fold(data, tr_tls, te_tls, n_ctx, seed=0, stop_epoch=None):
    """stop_epoch (inner folds): same cosine schedule, stopped after that many epochs, evaluated once at the end (k0 only)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    ti, pos, y = data.items(tr_tls)
    model = CtxDet(n_ctx).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CNN_EPOCHS)
    cnt = np.bincount(y, minlength=N_CLS)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (N_CLS * cnt), dtype=torch.float32, device=DEV))
    # held-out: stream windows (closeness, k = 0 and k = 2) + labelled negative / post-onset windows (binary CE for the epoch choice)
    eti, epos, ey = data.items(te_tls)
    is_stream = np.array([p < data.n_stream[te_tls[t]] for t, p in zip(eti, epos)])
    keep = (ey == 0) | (ey == N_CLS - 1)
    w_close = torch.tensor(np.minimum(np.arange(N_CLS), FIRST_POS) / FIRST_POS, dtype=torch.float32, device=DEV)
    out = []
    bs = max(BATCH // max(n_ctx // 4, 1), 32)
    n_ep = CNN_EPOCHS if stop_epoch is None else stop_epoch
    for ep in range(n_ep):
        model.train()
        perm = rng.permutation(len(y))
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]
            cut = rng.random(len(j)) < 0.5
            start = np.where(cut, pos[j] - rng.integers(0, n_ctx, len(j)), 0)
            idx, valid = data.context(tr_tls, ti[j], pos[j], n_ctx, np.maximum(start, 0))
            x, v = data.batch(idx, valid, g, augment=True)
            loss = ce(model(x, v), torch.as_tensor(y[j], device=DEV))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if stop_epoch is not None and ep < n_ep - 1:
            continue
        model.eval()
        res = {}
        with torch.no_grad():
            evals = (("k0", is_stream, None),) if stop_epoch is not None else (("k0", is_stream, None), ("k2", is_stream, 2), ("lab", keep, None))
            for name, sel, start in evals:
                ii = np.flatnonzero(sel)
                probs = []
                for b in range(0, len(ii), 1024):
                    jj = ii[b:b + 1024]
                    st = None if start is None else np.full(len(jj), start)
                    idx, valid = data.context(te_tls, eti[jj], epos[jj], n_ctx, st)
                    x, v = data.batch(idx, valid)
                    probs.append(torch.softmax(model(x, v), 1))
                res[name] = torch.cat(probs)
        if stop_epoch is not None:
            out.append(dict(close=(res["k0"] @ w_close).cpu().numpy(), p=res["k0"][:, FIRST_POS:].sum(1).cpu().numpy()))
            continue
        pe = res["lab"][:, FIRST_POS:].sum(1).cpu().numpy()
        yb = (ey[keep] == N_CLS - 1).astype(float)
        out.append(dict(close=(res["k0"] @ w_close).cpu().numpy(), close_k2=(res["k2"] @ w_close).cpu().numpy(),
                        p=res["k0"][:, FIRST_POS:].sum(1).cpu().numpy(),
                        ce=float(-np.mean(yb * np.log(pe + 1e-7) + (1 - yb) * np.log(1 - pe + 1e-7)))))
    run_fold.last_model = model  # final training (train_n2det_final.py detctx) keeps the weights
    # stream rows of the held-out recordings in data.items order -> row indices in d
    rows = np.concatenate([data.seq[tl][:data.n_stream[tl]] for tl in te_tls])
    return out, rows


if __name__ == "__main__":
    path = OUT / "n2det_ctx.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    d, _, post, m, mp = load_all()
    data = Data(d, post, m, mp)
    splits = unseen_splits(d)
    rem = d.target.to_numpy().astype(np.float64)
    tl_all = d.timeline.to_numpy()
    for n_ctx in [int(a) for a in sys.argv[1:]]:
        key = f"ctx{n_ctx}"
        if key in res:
            print(f"{key}: cached {res[key]['sanity']['spearman']:+.3f}", flush=True)
            continue
        t0 = time.perf_counter()
        runs = []
        for k, (tr, te) in enumerate(splits):
            t1 = time.perf_counter()
            runs.append(run_fold(data, list(dict.fromkeys(tl_all[tr])), list(dict.fromkeys(tl_all[te])), n_ctx))
            print(f"  {key} fold {k}: {time.perf_counter() - t1:.0f} s, CE curve " + " ".join(f"{e['ce']:.3f}" for e in runs[-1][0]), flush=True)
        C = np.array([[e["ce"] for e in r[0]] for r in runs])
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
        close, close2 = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for (out, rows), s in zip(runs, sel):
            close[rows] = out[s]["close"]
            close2[rows] = out[s]["close_k2"]
        assert not np.isnan(close).any()
        np.save(OUT / f"n2det_ctx{n_ctx}_oof.npy", close.astype(np.float32))
        res[key] = dict(sanity=sanity(close, rem), sanity_start_k2=sanity(close2, rem), epochs=[s + 1 for s in sel],
                        time_s=time.perf_counter() - t0)
        path.write_text(json.dumps(res, indent=2, default=float))
        s, s2 = res[key]["sanity"], res[key]["sanity_start_k2"]
        print(f"{key}: Spearman {s['spearman']:+.3f} AUC60 {s['auc_last60_vs_300plus']:.3f} AUC30 {s['auc_last30_vs_60_150']:.3f} | "
              f"stream from window 2: Spearman {s2['spearman']:+.3f} | epochs {res[key]['epochs']} ({res[key]['time_s']:.0f} s) | "
              f"ref -0.521 / 0.951 / 0.821; stop rule Spearman <= -0.551: {'PASS' if s['spearman'] <= -0.551 else 'STOP'}", flush=True)
