"""Step D: small causal GRU on per-window features -> distribution over the 11 B classes, dual CV.

Design: docs/gru_design.md.
Inputs per window (125): 62 stateless window features (base 40 + spec 20 + eye 2), the same 62 minus
the recording's first-2-min running mean (causal, as CausalFeatureStream *_rel), log1p(elapsed_s)
clipped at 1500 s. Standardised with the training fold's mean / std.
Model: Linear(125->64) + GELU + LayerNorm -> GRU(64, 2 layers, causal) -> Linear(64->11).
Variants:
  "w"  : CE weighted with the metric's window weights; decision = continuous median with omega = 1
         (the weighting is already inside the learned distribution).
  "uw" : unweighted CE; decision with omega (as dist_hgb B).
Right padding + causal GRU: outputs never see later windows, the recording length or its end.
Per fold: SEEDS models; probabilities averaged over seeds at every checkpoint (every EVAL_EVERY
epochs); checkpoint of fold k = argmin of the mean curve of the other folds (LOFO).

Usage: python scripts/gru_cv.py [w|uw ...]   (default: w uw)
"""
import json
import sys
import time

import numpy as np
import torch
from torch import nn

from ablation_c import build_table_ext
from dist_hgb import VARIANTS, class_stats, decide, to_class
from dual_cv import OUT, dual_report, fmt, seen_splits, unseen_splits
from report_bins import contributions
from stream_utils import RANGE_NAMES, summarize, window_weights
from train_hgb import SAVE

EDGES = VARIANTS["B"]
SEEDS = (0, 1, 2)
EPOCHS = 150
EVAL_EVERY = 5
BATCH = 32
LR, WD = 2e-3, 1e-2
HIDDEN = 64
BASELINE_N = 24  # windows within the first 120 s (elapsed = n * 5 s <= 120 s)
ELAPSED_CLIP = 1500.0
GAIN_SD = 0.1  # log10 amplitude gain per recording and channel (augmentation)
NOISE_SD = 0.05
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_inputs():
    """-> d (train+val windows, original row order), list of per-recording index arrays in time order,
    window feature matrix (N, 62) and the column offsets for the gain augmentation."""
    df, _, _ = build_table_ext()  # meta + targets, rows aligned with features_base.npz
    zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
    F = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)  # 40 + 22
    n_ch, n_pc = 4, 10
    # gain g (log10) per channel: +2g on 5 band powers and var, +g on linelen and maxabs, 0 elsewhere
    gain = np.zeros((n_ch, F.shape[1]), np.float32)
    for c in range(n_ch):
        gain[c, c * n_pc + np.array([0, 1, 2, 3, 4, 7])] = 2.0
        gain[c, c * n_pc + np.array([8, 9])] = 1.0
    m = (df.split != "test").to_numpy()
    d, F = df[m].reset_index(drop=True), F[m]
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    return d, recs, F, gain


def seq_inputs(Fw):
    """(L, 62) window features of one recording -> (L, 125) causal inputs."""
    L = len(Fw)
    n = np.minimum(np.arange(1, L + 1), BASELINE_N)
    base = np.cumsum(Fw, axis=0)[n - 1] / n[:, None]
    el = np.log1p(np.minimum(np.arange(1, L + 1) * 5.0, ELAPSED_CLIP))[:, None]
    return np.concatenate([Fw, Fw - base, el], axis=1).astype(np.float32)


class GRUNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=HIDDEN):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, h=None):
        y, h = self.gru(self.inp(x), h)
        return self.out(y), h


def pad(seqs, fill=0.0):
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L) + seqs[0].shape[1:], fill, dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def run_fold(d, recs, F, gain, tr, te, variant, seed):
    """Train one model; return list over checkpoints of (n_te_windows, 11) probs (rows in d[te] order)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_recs = [r for r in recs if tr[r[0]]]
    te_recs = [r for r in recs if te[r[0]]]
    dtr = d[tr]
    w_all = np.zeros(len(d), np.float32)
    w_all[np.flatnonzero(tr)] = window_weights(dtr.reset_index(drop=True)) if variant == "w" else 1.0
    y_all = to_class(d.target.to_numpy(), EDGES)
    X_tr = [seq_inputs(F[r]) for r in tr_recs]
    allx = np.concatenate(X_tr)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    gain_std = np.concatenate([gain, np.zeros((4, F.shape[1] + 1), np.float32)], axis=1) / sd  # rel/elapsed: 0
    Xte = torch.tensor(pad([(seq_inputs(F[r]) - mu) / sd for r in te_recs]), device=DEV)
    model = GRUNet(allx.shape[1], len(EDGES) - 1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xs = [(x - mu) / sd for x in X_tr]
    Y = [y_all[r] for r in tr_recs]
    W = [w_all[r] for r in tr_recs]
    ce = nn.CrossEntropyLoss(reduction="none")
    te_pos = np.concatenate([np.searchsorted(np.flatnonzero(te), r) for r in te_recs])
    out = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(Xs))
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            xb = []
            for i in idx:  # augmentation: per-recording channel gains (raw part only; rel cancels) + noise
                g = rng.normal(0, GAIN_SD, 4).astype(np.float32) @ gain_std
                xb.append(Xs[i] + g + rng.normal(0, NOISE_SD, Xs[i].shape).astype(np.float32))
            x = torch.tensor(pad(xb), device=DEV)
            y = torch.tensor(pad([Y[i][:, None] for i in idx], -1)[..., 0], device=DEV).long()
            w = torch.tensor(pad([W[i][:, None] for i in idx])[..., 0], device=DEV)
            logits, _ = model(x)
            loss = (ce(logits.reshape(-1, logits.shape[-1]), y.clamp(min=0).reshape(-1)) * w.reshape(-1)).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                P = torch.softmax(model(Xte)[0], -1).cpu().numpy()
            flat = np.concatenate([P[i, :len(r)] for i, r in enumerate(te_recs)])
            probs = np.empty_like(flat)
            probs[te_pos] = flat
            out.append(probs)
    return out


def run_protocol(d, recs, F, gain, splits, variant):
    curves, preds, probas = [], [], []
    for k, (tr, te) in enumerate(splits):
        t0 = time.perf_counter()
        dte = d[te].reset_index(drop=True)
        runs = [run_fold(d, recs, F, gain, tr, te, variant, s) for s in SEEDS]
        _, omega = class_stats(d[tr].reset_index(drop=True), EDGES)
        om = np.ones_like(omega) if variant == "w" else omega
        Pk = [np.mean([r[j] for r in runs], axis=0) for j in range(len(runs[0]))]
        q = [decide(P.astype(np.float64), None, om, True, EDGES).astype(np.float32) for P in Pk]
        curves.append(np.array([summarize(dte, x)["wbmae"] for x in q]))
        preds.append(q)
        probas.append(Pk)
        print(f"    fold {k}: best epoch {EVAL_EVERY * (curves[-1].argmin() + 1)} "
              f"{curves[-1].min():.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    C = np.stack(curves)
    sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
    return ([preds[k][j] for k, j in enumerate(sel)], [probas[k][j] for k, j in enumerate(sel)],
            [EVAL_EVERY * (j + 1) for j in sel], C)


if __name__ == "__main__":
    variants = sys.argv[1:] or ["w", "uw"]
    d, recs, F, gain = load_inputs()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    print(f"device {DEV}; {len(recs)} recordings, {len(d)} windows, inputs {F.shape[1] * 2 + 1}", flush=True)
    summary_path = OUT / "gru.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for v in variants:
        t0 = time.perf_counter()
        preds, info = {}, {}
        for proto, splits in protocols.items():
            print(f"=== GRU-{v} {proto}", flush=True)
            preds[proto], probas, ep, C = run_protocol(d, recs, F, gain, splits, v)
            info[f"epochs_{proto}"] = ep
            info[f"curve_mean_{proto}"] = C.mean(0).round(3).tolist()
            np.savez(OUT / f"gru_{v}_{proto}_preds.npz", *preds[proto])
            np.savez(OUT / f"gru_{v}_{proto}_proba.npz", *probas)
        res = dual_report(f"GRU-{v}", d, protocols, preds, dict(seeds=list(SEEDS), **info,
                                                                time_s=time.perf_counter() - t0))
        res["contrib_plain"] = {p: dict(zip(RANGE_NAMES, contributions(d, protocols[p], preds[p])[0].tolist()))
                                for p in protocols}
        summary[f"GRU-{v}"] = res
        summary_path.write_text(json.dumps(summary, indent=2))
        print(fmt(res), f"({res['time_s']:.0f} s, epochs {info['epochs_unseen']} / {info['epochs_seen']})", flush=True)
        for p in protocols:
            print(f"    {p:6s} contrib: " + "  ".join(f"{k} {x:.2f}" for k, x in res["contrib_plain"][p].items()),
                  flush=True)
