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

Configs (CONFIGS): "w" / "uw" = first runs (150 epochs, lr 2e-3, eval every 5); "u64_d2" ... =
second round (30 epochs, lr 1e-3, eval every epoch, dropout / weight decay / hidden size varied).
Dropout is applied after the input projection and between the GRU layers.

Usage: python scripts/gru_cv.py [config ...]   (default: w uw)
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
BASE_CFG = dict(weighted=True, epochs=EPOCHS, eval_every=EVAL_EVERY, lr=LR, wd=WD, hidden=HIDDEN,
                dropout=0.2, in_dropout=0.0)
CONFIGS = {
    "w": dict(BASE_CFG),
    "uw": dict(BASE_CFG, weighted=False),
    "u64_d2": dict(BASE_CFG, weighted=False, epochs=30, eval_every=1, lr=1e-3, in_dropout=0.2),
    "u64_d4": dict(BASE_CFG, weighted=False, epochs=30, eval_every=1, lr=1e-3, dropout=0.4, in_dropout=0.4, wd=5e-2),
    "u32_d2": dict(BASE_CFG, weighted=False, epochs=30, eval_every=1, lr=1e-3, hidden=32, in_dropout=0.2),
    "u32_d4": dict(BASE_CFG, weighted=False, epochs=30, eval_every=1, lr=1e-3, hidden=32, dropout=0.4,
                   in_dropout=0.4, wd=5e-2),
    "w64_d4": dict(BASE_CFG, epochs=30, eval_every=1, lr=1e-3, dropout=0.4, in_dropout=0.4, wd=5e-2),
}
# seed-count study (scripts/seeds_cv.py): uw, trained only to epoch 40 (selection was always 10-15)
for _n in (3, 5, 10):
    CONFIGS[f"uw_n{_n}"] = dict(CONFIGS["uw"], run_epochs=40, n_seeds=_n)
# experiment 3 (scripts/seq_cv.py): other sequence models, uw settings, 10 seeds, up to 30 epochs
for _a in ("lstm", "tcn", "tcn_long"):
    CONFIGS[f"{_a}_n10"] = dict(CONFIGS["uw"], arch=_a, run_epochs=30, n_seeds=10)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_inputs(include_test=False):
    """-> d (train+val windows [+ test if include_test], original row order), list of per-recording index arrays in time order,
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
    m = np.ones(len(df), bool) if include_test else (df.split != "test").to_numpy()
    d, F = df[m].reset_index(drop=True), F[m]
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
    return d, recs, F, gain


def seq_inputs(Fw, Ew=None):
    """(L, 62) window features of one recording -> (L, 125) causal inputs.
    Ew (L, E) optional per-window extras (e.g. CNN embeddings): appended as Ew and Ew - first-2-min mean."""
    L = len(Fw)
    n = np.minimum(np.arange(1, L + 1), BASELINE_N)
    base = np.cumsum(Fw, axis=0)[n - 1] / n[:, None]
    el = np.log1p(np.minimum(np.arange(1, L + 1) * 5.0, ELAPSED_CLIP))[:, None]
    parts = [Fw, Fw - base, el]
    if Ew is not None:
        parts += [Ew, Ew - np.cumsum(Ew, axis=0)[n - 1] / n[:, None]]
    return np.concatenate(parts, axis=1).astype(np.float32)


class GRUNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=HIDDEN, dropout=0.2, in_dropout=0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(in_dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, h=None):
        y, h = self.gru(self.inp(x), h)
        return self.out(y), h


class LSTMNet(GRUNet):
    """Same as GRUNet with a 2-layer LSTM (experiment 3)."""

    def __init__(self, n_in, n_out, hidden=HIDDEN, dropout=0.2, in_dropout=0.0):
        super().__init__(n_in, n_out, hidden, dropout, in_dropout)
        self.gru = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)


class TCNNet(nn.Module):
    """Causal dilated temporal convolutions (left padding only): kernel 3, dilations 1..32,
    receptive field 127 windows (10.6 min). Residual blocks with GELU, dropout, LayerNorm."""

    def __init__(self, n_in, n_out, hidden=HIDDEN, dropout=0.2, in_dropout=0.0, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(in_dropout))
        self.dil = dilations
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, dilation=k) for k in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in dilations)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, h=None):
        z = self.inp(x)  # (B, L, H)
        for k, conv, norm in zip(self.dil, self.convs, self.norms):
            y = conv(nn.functional.pad(z.transpose(1, 2), (2 * k, 0))).transpose(1, 2)
            z = norm(z + self.drop(self.act(y)))
        return self.out(z), None


class TCNLongNet(TCNNet):
    """TCN with dilations 1..128: receptive field 511 windows (42.6 min) >= the longest recording."""

    def __init__(self, n_in, n_out, hidden=HIDDEN, dropout=0.2, in_dropout=0.0):
        super().__init__(n_in, n_out, hidden, dropout, in_dropout, dilations=(1, 2, 4, 8, 16, 32, 64, 128))


ARCHS = {"gru": GRUNet, "lstm": LSTMNet, "tcn": TCNNet, "tcn_long": TCNLongNet}


def pad(seqs, fill=0.0):
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L) + seqs[0].shape[1:], fill, dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def run_fold(d, recs, F, gain, tr, te, cfg, seed, stop_epoch=None, extra=None):
    """Train one model; return list over checkpoints of (n_te_windows, 11) probs (rows in d[te] order).
    te=None (final training): same schedule (cosine over cfg["epochs"]), stop after ``stop_epoch``
    epochs, no evaluation; returns (model, mu, sd).
    extra: optional (len(d), E) per-window inputs appended via seq_inputs (no gain augmentation)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_recs = [r for r in recs if tr[r[0]]]
    te_recs = [r for r in recs if te[r[0]]] if te is not None else []
    dtr = d[tr]
    w_all = np.zeros(len(d), np.float32)
    w_all[np.flatnonzero(tr)] = window_weights(dtr.reset_index(drop=True)) if cfg["weighted"] else 1.0
    y_all = to_class(d.target.to_numpy(), EDGES)

    def inputs(r):
        return seq_inputs(F[r], None if extra is None else extra[r])

    X_tr = [inputs(r) for r in tr_recs]
    allx = np.concatenate(X_tr)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    n_zero = allx.shape[1] - gain.shape[1]  # rel / elapsed / extra columns: no gain augmentation
    gain_std = np.concatenate([gain, np.zeros((4, n_zero), np.float32)], axis=1) / sd
    Xte = torch.tensor(pad([(inputs(r) - mu) / sd for r in te_recs]), device=DEV) if te_recs else None
    model = ARCHS[cfg.get("arch", "gru")](allx.shape[1], len(EDGES) - 1, cfg["hidden"], cfg["dropout"],
                                          cfg["in_dropout"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    Xs = [(x - mu) / sd for x in X_tr]
    Y = [y_all[r] for r in tr_recs]
    W = [w_all[r] for r in tr_recs]
    ce = nn.CrossEntropyLoss(reduction="none")
    te_pos = np.concatenate([np.searchsorted(np.flatnonzero(te), r) for r in te_recs]) if te_recs else None
    out = []
    for ep in range(1, cfg.get("run_epochs", cfg["epochs"]) + 1):
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
        if te is None:
            if ep == stop_epoch:
                return model.eval(), mu, sd
            continue
        if ep % cfg["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                P = torch.softmax(model(Xte)[0], -1).cpu().numpy()
            flat = np.concatenate([P[i, :len(r)] for i, r in enumerate(te_recs)])
            probs = np.empty_like(flat)
            probs[te_pos] = flat
            out.append(probs)
    return out


def run_protocol(d, recs, F, gain, splits, cfg):
    curves, preds, probas = [], [], []
    for k, (tr, te) in enumerate(splits):
        t0 = time.perf_counter()
        dte = d[te].reset_index(drop=True)
        runs = [run_fold(d, recs, F, gain, tr, te, cfg, s) for s in SEEDS]
        _, omega = class_stats(d[tr].reset_index(drop=True), EDGES)
        om = np.ones_like(omega) if cfg["weighted"] else omega
        Pk = [np.mean([r[j] for r in runs], axis=0) for j in range(len(runs[0]))]
        q = [decide(P.astype(np.float64), None, om, True, EDGES).astype(np.float32) for P in Pk]
        curves.append(np.array([summarize(dte, x)["wbmae"] for x in q]))
        preds.append(q)
        probas.append(Pk)
        print(f"    fold {k}: in-fold best epoch (diagnostic only) {cfg['eval_every'] * (curves[-1].argmin() + 1)} "
              f"{curves[-1].min():.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    C = np.stack(curves)
    sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
    return ([preds[k][j] for k, j in enumerate(sel)], [probas[k][j] for k, j in enumerate(sel)],
            [cfg["eval_every"] * (j + 1) for j in sel], C)


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
            preds[proto], probas, ep, C = run_protocol(d, recs, F, gain, splits, CONFIGS[v])
            info[f"epochs_{proto}"] = ep
            info[f"curve_mean_{proto}"] = C.mean(0).round(3).tolist()
            np.savez(OUT / f"gru_{v}_{proto}_preds.npz", *preds[proto])
            np.savez(OUT / f"gru_{v}_{proto}_proba.npz", *probas)
        res = dual_report(f"GRU-{v}", d, protocols, preds, dict(seeds=list(SEEDS), cfg=CONFIGS[v], **info,
                                                                time_s=time.perf_counter() - t0))
        res["contrib_plain"] = {p: dict(zip(RANGE_NAMES, contributions(d, protocols[p], preds[p])[0].tolist()))
                                for p in protocols}
        summary[f"GRU-{v}"] = res
        summary_path.write_text(json.dumps(summary, indent=2))
        print(fmt(res), f"({res['time_s']:.0f} s, epochs {info['epochs_unseen']} / {info['epochs_seen']})", flush=True)
        for p in protocols:
            print(f"    {p:6s} contrib: " + "  ".join(f"{k} {x:.2f}" for k, x in res["contrib_plain"][p].items()),
                  flush=True)
