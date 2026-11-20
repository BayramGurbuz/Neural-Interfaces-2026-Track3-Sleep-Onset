"""Work L (20 Nov, research path B): end-to-end raw-EEG CNN -> GRU, trained jointly on recording sequences.

Per window: the CNN encoder of cnn_encoder.py (35k parameters, raw (4, 640) uV scaled x / 20, clipped +-15) -> 32-d embedding E.
Per recording: GRU-uw (GRUNet, 2 x 64) on [the 125 causal hand-feature inputs (seq_inputs: 62 features, difference to the first-2-min
mean, log elapsed), E, E - mean(E over the first min(t, 24) windows)], 11-class B distribution, unweighted CE (+ 0.3 x CE of the
encoder's own window head). This is exactly the input layout of the deployed GRU-emb component, so a trained (encoder, GRU) pair is
served by the existing wrapper path (cnn_states / gemb_states, gemb_mu = [mu_125, 0], gemb_sd = [sd_125, 1]); only the training is new
(joint instead of two-stage, which drifted: the GRU over-trusted in-sample embeddings).

Augmentation as GRU ctx: stream-start crop (0.5 none / 0.35 1-12 windows / 0.15 wide), features and baselines rebuilt from the crop;
per-recording channel gain g (log10, sd 0.1) applied consistently to the raw windows (x 10^g) and the features (g @ gain matrix), noise.
Deterministic GPU settings (repro.set_deterministic must be called by the entry script before CUDA work).
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from cnn_encoder import CLIP, SCALE, Encoder
from dist_hgb import to_class
from gru_cv import BASELINE_N, DEV, EDGES, GRUNet, seq_inputs

EMB = 32
CFG = dict(lr=1.5e-3, wd=1e-2, batch=16, aux=0.3, gain_sd=0.1, noise_feat=0.05, noise_raw=0.05, crop=(0.5, 0.35, 0.15),
           hidden=64, dropout=0.2, epochs=18, checkpoints=(6, 10, 14, 18))


class E2ENet(nn.Module):
    def __init__(self, n_feat, n_out=len(EDGES) - 1, emb=EMB, hidden=64, dropout=0.2):
        super().__init__()
        self.enc = Encoder(n_out, emb)
        self.gru = GRUNet(n_feat + 2 * emb, n_out, hidden, dropout)
        self.emb = emb

    def forward(self, feats, raw_list):
        """feats (B, T, n_feat) standardised hand inputs (padded); raw_list: list of (L_b, 4, 640) prepped raw windows.
        -> GRU logits (B, T, n_out), encoder window logits (sum L_b, n_out) in the order of raw_list."""
        lens = [len(r) for r in raw_list]
        lo_w, e = self.enc(torch.cat(raw_list))
        E = pad_sequence(list(torch.split(e, lens)), batch_first=True)  # (B, T, emb), zeros after each recording's end
        T = feats.shape[1]
        if E.shape[1] < T:
            E = torch.cat([E, E.new_zeros(E.shape[0], T - E.shape[1], E.shape[2])], 1)
        n = torch.clamp(torch.arange(1, T + 1, device=feats.device), max=BASELINE_N)
        base = torch.cumsum(E, 1)[:, n - 1] / n[None, :, None].to(E.dtype)
        y, _ = self.gru(torch.cat([feats, E, E - base], -1))
        return y, lo_w


def prep(x):
    return torch.clamp(x / SCALE, -CLIP, CLIP)


def train_e2e(d, recs, F, gain, X, tr, seed, cfg=CFG, on_checkpoint=None):
    """d / recs / F / gain as gru_cv.load_inputs; X: (len(d), 4, 640) float32 raw windows on DEV, rows aligned with d; tr: bool mask.
    on_checkpoint(epoch, net, mu, sd) is called after every epoch in cfg["checkpoints"] (net in eval mode, restored to train after).
    Returns (net, mu, sd) after cfg["epochs"] epochs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_recs = [r for r in recs if tr[r[0]]]
    y_all = torch.tensor(to_class(d.target.to_numpy(), EDGES), device=DEV)
    X_tr = [seq_inputs(F[r]) for r in tr_recs]
    allx = np.concatenate(X_tr)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    gain_std = np.concatenate([gain, np.zeros((4, allx.shape[1] - gain.shape[1]), np.float32)], axis=1) / sd
    net = E2ENet(allx.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    ce = nn.CrossEntropyLoss(reduction="none")
    crop = cfg["crop"]

    def crop_k(n_):
        u = rng.random()
        if u < crop[0]:
            return 0
        if u < crop[0] + crop[1] or n_ - 12 < 13:
            return int(rng.integers(1, 13))
        return int(rng.integers(13, n_ - 12 + 1))

    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        order = rng.permutation(len(tr_recs))
        for b0 in range(0, len(order), cfg["batch"]):
            feats, raws, ys = [], [], []
            for i in order[b0:b0 + cfg["batch"]]:
                r = tr_recs[i]
                k = crop_k(len(r))
                rows = r[k:]
                g = rng.normal(0, cfg["gain_sd"], 4).astype(np.float32)
                xi = (X_tr[i] - mu) / sd if k == 0 else (seq_inputs(F[rows]) - mu) / sd
                feats.append(xi + g @ gain_std + rng.normal(0, cfg["noise_feat"], xi.shape).astype(np.float32))
                ridx = torch.as_tensor(rows, device=DEV)
                xr = X[ridx] * torch.as_tensor(10.0 ** g, device=DEV)[None, :, None]
                xr = prep(xr)
                xr = xr + cfg["noise_raw"] * torch.randn(xr.shape, device=DEV)
                raws.append(xr)
                ys.append(y_all[ridx])
            lens = [len(x) for x in feats]
            fb = torch.tensor(np.stack([np.pad(x, ((0, max(lens) - len(x)), (0, 0))) for x in feats]), device=DEV)
            out, lo_w = net(fb, raws)
            yb = pad_sequence(ys, batch_first=True, padding_value=-1)
            mask = yb >= 0
            loss_seq = ce(out[mask], yb[mask]).mean()
            loss_w = ce(lo_w, torch.cat(ys)).mean()
            loss = loss_seq + cfg["aux"] * loss_w
            opt.zero_grad()
            loss.backward()
            opt.step()
        if on_checkpoint is not None and ep in cfg["checkpoints"]:
            net.eval()
            on_checkpoint(ep, net, mu, sd)
    net.eval()
    return net, mu, sd


@torch.no_grad()
def predict_e2e(net, mu, sd, F, X, rows_list, chunk=16):
    """rows_list: list of window-index arrays (streams, time order; inputs restart at rows[0]) -> concatenated (sum L, n_out) probs."""
    net.eval()
    out = []
    for c0 in range(0, len(rows_list), chunk):
        part = rows_list[c0:c0 + chunk]
        feats = [(seq_inputs(F[r]) - mu) / sd for r in part]
        lens = [len(x) for x in feats]
        fb = torch.tensor(np.stack([np.pad(x, ((0, max(lens) - len(x)), (0, 0))) for x in feats]), device=DEV)
        raws = [prep(X[torch.as_tensor(r, device=DEV)]) for r in part]
        y, _ = net(fb, raws)
        P = torch.softmax(y, -1).cpu().numpy()
        out += [P[b, :n_] for b, n_ in enumerate(lens)]
    return np.concatenate(out)
