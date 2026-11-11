"""Stage 1 of the two-stage CNN experiment: a small 1D CNN window encoder trained on the raw
(4, 640) microvolt windows with the 11-class B distribution loss (unweighted CE, as GRU-uw),
dual CV, one encoder per fold and seed (held-out recordings never seen by their encoder).

Receptive field: 4 stride-2 blocks (kernel 7) + 2 dilated blocks (dilation 2, 4) = 667 samples >= 640,
so slow eye movements (0.2-0.6 Hz) fit. ~35k parameters. Per-window scaling only (x / 100, clipped):
no recording-level statistics.

Outputs (OUT = ~/neuralbench/save/cv/dual):
  cnn_<name>_{proto}_{preds,proba}.npz     held-out decisions / seed-mean probabilities (LOFO epoch)
  cnn_<name>_{proto}_f<k>_s<seed>.npz      emb (N, 32) and logits (N, 11) for ALL train+val windows from
                                           the fold-k seed-s encoder (training rows in-sample, held-out
                                           rows out-of-sample) for the GRU stage, plus epoch, ce_in, ce_out
  cnn_<name>.json                          summary (epochs, CE in/out per fold, dual CV)

Mode "half" (stacking variant): per fold and seed two encoders, each fit on one half of the training
recordings (split by recording); training rows get the logits of the encoder that did not see them,
held-out rows the mean of both. Feed "logits" (aligned across encoders) to the GRU, not "emb".

Usage: python scripts/cnn_encoder.py [name] [n_seeds] [full|half]   (default cnn3, 3, full)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dist_hgb import class_stats, decide, to_class
from dual_cv import OUT, seen_splits, unseen_splits
from gru_cv import DEV, EDGES, load_inputs
from range_breakdown import breakdown, load_preds, print_breakdown
from stream_utils import summarize

RAW = Path.home() / "neuralbench" / "save" / "raw_windows.npy"
SCALE, CLIP = 20.0, 15.0  # median |x| 5.4 uV -> 0.27; clip at +-300 uV
EPOCHS, BATCH, LR, WD = 20, 256, 1e-3, 1e-2
GAIN_SD, NOISE_SD = 0.1, 0.1  # log10 gain per window and channel; noise in scaled units (2 uV)
EMB = 32


class Encoder(nn.Module):
    def __init__(self, n_out=11, emb=EMB, dropout=0.1):
        super().__init__()
        spec = [(4, 16, 2, 1), (16, 32, 2, 1), (32, 32, 2, 1), (32, 32, 2, 1), (32, 32, 1, 2), (32, 32, 1, 4)]
        layers = []
        for ci, co, s, dil in spec:
            layers += [nn.Conv1d(ci, co, 7, stride=s, dilation=dil, padding=3 * dil), nn.GELU(), nn.Dropout(dropout)]
        self.conv = nn.Sequential(*layers)
        self.emb = nn.Sequential(nn.Linear(64, emb), nn.GELU())
        self.out = nn.Linear(emb, n_out)

    def forward(self, x):
        z = self.conv(x)  # (B, 32, 40)
        e = self.emb(torch.cat([z.mean(2), z.amax(2)], 1))
        return self.out(e), e


def receptive_field(model):
    rf, jump = 1, 1
    for m in model.conv:
        if isinstance(m, nn.Conv1d):
            rf += (m.kernel_size[0] - 1) * m.dilation[0] * jump
            jump *= m.stride[0]
    return rf


def prep(x):
    return torch.clamp(x / SCALE, -CLIP, CLIP)


@torch.no_grad()
def infer(model, X, idx, bs=2048):
    model.eval()
    logits, embs = [], []
    for b in range(0, len(idx), bs):
        lo, e = model(prep(X[idx[b:b + bs]]))
        logits.append(lo)
        embs.append(e)
    return torch.cat(logits), torch.cat(embs)


def train_fold(X, y, tr, te, seed):
    """Returns per-epoch held-out probs, per-epoch (ce_in, ce_out), and per-epoch state dicts."""
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    tr_idx = torch.as_tensor(np.flatnonzero(tr), device=DEV)
    te_idx = torch.as_tensor(np.flatnonzero(te), device=DEV)
    sub = tr_idx[torch.randperm(len(tr_idx), generator=g, device=DEV)[:len(te_idx)]]  # in-sample CE subset
    model = Encoder().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    ce = nn.CrossEntropyLoss()
    probs, ces, states = [], [], []
    for ep in range(EPOCHS):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g, device=DEV)]
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            x = X[idx]
            gain = torch.pow(10.0, GAIN_SD * torch.randn(len(idx), 4, 1, generator=g, device=DEV))
            x = prep(x * gain) + NOISE_SD * torch.randn(x.shape, generator=g, device=DEV)
            loss = ce(model(x)[0], y[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        lo_te, _ = infer(model, X, te_idx)
        lo_in, _ = infer(model, X, sub)
        probs.append(torch.softmax(lo_te, 1).cpu().numpy())
        ces.append((float(ce(lo_in, y[sub])), float(ce(lo_te, y[te_idx]))))
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    return probs, ces, states


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "cnn3"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mode = sys.argv[3] if len(sys.argv) > 3 else "full"  # "half": split-half encoders (stacking variant)
    d, recs, F, gain = load_inputs()
    mask = (np.load(Path.home() / "neuralbench" / "save" / "features_base.npz")["split"] != "test")
    X = torch.as_tensor(np.load(RAW, mmap_mode="r")[mask], device=DEV)
    y = torch.as_tensor(to_class(d.target.to_numpy(), EDGES), device=DEV).long()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    m0 = Encoder()
    print(f"device {DEV}; {len(d)} windows on GPU ({X.numel() * 4 / 1e6:.0f} MB); encoder params "
          f"{sum(p.numel() for p in m0.parameters())}, receptive field {receptive_field(m0)} samples", flush=True)
    summary = {"name": name, "n_seeds": n_seeds, "mode": mode, "epochs": {}, "ce": {}}
    preds = {}
    t_all = time.perf_counter()
    tl = d.timeline.to_numpy()
    for proto, splits in protocols.items():
        curves, cands, probas, fold_runs, fold_halves = [], [], [], [], []
        for k, (tr, te) in enumerate(splits):
            t0 = time.perf_counter()
            if mode == "half":  # two encoders per seed, each fit on one half of the training recordings
                rec_tr = np.unique(tl[tr])
                hA = np.isin(tl, np.random.default_rng(1000 + k).permutation(rec_tr)[:len(rec_tr) // 2]) & tr
                halves = (hA, tr & ~hA)
                runs = [[train_fold(X, y, h, te, s) for h in halves] for s in range(n_seeds)]
                # held-out probs / CE: mean of the two encoders (CE out of the mean logits approximated by mean CE)
                runs = [(list(np.mean([r[0] for r in pair], axis=0)),
                         [tuple(np.mean([r[1][j] for r in pair], axis=0)) for j in range(EPOCHS)],
                         [r[2] for r in pair]) for pair in runs]
            else:
                halves = None
                runs = [train_fold(X, y, tr, te, s) for s in range(n_seeds)]
            fold_halves.append(halves)
            dte = d[te].reset_index(drop=True)
            _, omega = class_stats(d[tr].reset_index(drop=True), EDGES)
            Pk = [np.mean([r[0][j] for r in runs], axis=0) for j in range(EPOCHS)]
            q = [decide(P.astype(np.float64), None, omega, True, EDGES).astype(np.float32) for P in Pk]
            curves.append([summarize(dte, x)["wbmae"] for x in q])
            cands.append(q)
            probas.append(Pk)
            fold_runs.append(runs)
            ce_out = np.mean([[c[1] for c in r[1]] for r in runs], axis=0)
            print(f"  {proto} fold {k}: {time.perf_counter() - t0:.0f} s, W-bMAE curve " +
                  " ".join(f"{x:.1f}" for x in curves[-1]) + "\n      CE out curve " +
                  " ".join(f"{x:.3f}" for x in ce_out), flush=True)
        C = np.array(curves)
        sel = [int(np.delete(C, k, axis=0).mean(0).argmin()) for k in range(len(splits))]
        preds[proto] = [cands[k][j] for k, j in enumerate(sel)]
        summary["epochs"][proto] = [j + 1 for j in sel]
        np.savez(OUT / f"cnn_{name}_{proto}_preds.npz", *preds[proto])
        np.savez(OUT / f"cnn_{name}_{proto}_proba.npz", *[probas[k][j] for k, j in enumerate(sel)])
        summary["ce"][proto] = []
        for k, ((tr, te), runs, halves, j) in enumerate(zip(splits, fold_runs, fold_halves, sel)):
            ce_in = float(np.mean([r[1][j][0] for r in runs]))
            ce_out = float(np.mean([r[1][j][1] for r in runs]))
            summary["ce"][proto].append(dict(epoch=j + 1, ce_in=ce_in, ce_out=ce_out, ratio=ce_out / ce_in))
            model = Encoder().to(DEV)
            all_idx = torch.arange(len(d), device=DEV)
            for s, r in enumerate(runs):
                if halves is None:
                    model.load_state_dict(r[2][j])
                    lo, e = infer(model, X, all_idx)
                else:  # training rows: logits of the encoder that did not see them; held-out: mean of both
                    outs = []
                    for st in r[2]:
                        model.load_state_dict(st[j])
                        outs.append(infer(model, X, all_idx))
                    lo = (outs[0][0] + outs[1][0]) / 2
                    e = (outs[0][1] + outs[1][1]) / 2
                    hA, hB = (torch.as_tensor(h, device=DEV) for h in halves)
                    lo[hA], lo[hB] = outs[1][0][hA], outs[0][0][hB]
                    e[hA], e[hB] = outs[1][1][hA], outs[0][1][hB]
                np.savez(OUT / f"cnn_{name}_{proto}_f{k}_s{s}.npz", emb=e.cpu().numpy(), logits=lo.cpu().numpy(),
                         epoch=j + 1, ce_in=r[1][j][0], ce_out=r[1][j][1])
            print(f"  {proto} fold {k}: LOFO epoch {j + 1}, CE in {ce_in:.3f} out {ce_out:.3f} "
                  f"(out/in {ce_out / ce_in:.3f})", flush=True)
    summary["time_s"] = time.perf_counter() - t_all
    b = breakdown(d, protocols, preds)
    summary["dual"] = float(b["macro"]["wbmae"])
    summary["unseen"], summary["seen"] = float(b["unseen"]["wbmae"]), float(b["seen"]["wbmae"])
    (OUT / f"cnn_{name}.json").write_text(json.dumps(summary, indent=2))
    for ref_name in ("gru_uw_n3", "gru_uw_n10"):
        print_breakdown(f"CNN {name} alone vs {ref_name}", b, breakdown(d, protocols, load_preds(ref_name, protocols)))
    print(f"total {summary['time_s']:.0f} s; epochs {summary['epochs']}", flush=True)
