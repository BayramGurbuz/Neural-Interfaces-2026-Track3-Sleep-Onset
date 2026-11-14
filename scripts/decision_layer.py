"""Decision-layer experiments on the mix3 recipe (no retraining): exact metric weights, 30 s onset grid,
countdown smoothing and their combinations, dual CV on the saved out-of-fold class probabilities.

Components per fold (as scripts/mix_seq.py): three sequence models (gru_uw_n10, gru_tcn_n10, gru_cnn3_emb;
B-class probabilities) and the HGB ensemble (c) (4 members averaged on the common grid). Every
distribution is refined to 1 s cells of remaining time y in (0, 600) plus the point class y = 600, and
decisions are taken on 1201 "slots": atoms at y = 0, 1, ..., 600 interleaved with the 600 unit cells.

weights
  "old"   : class weights as in the submitted model (omega_B for the sequence models, grid omega for HGB).
  "exact" : w(t, y) = W_bin(y) / (n(D) * S(D)), D = t + y: in a recording whose onset is D the windows end
            at 5, 10, ..., floor(D / 5) * 5; S(D) = sum of the weights of its non-empty ranges, n(D) = number
            of its windows in the range of y. Point class (clipped, D >= t + 600): expectation over the
            in-fold training recordings with D >= t + 600 (none: D = t + 600).
structure
  "comp"  : as the reference: each component weighted and normalised, sequence models averaged, mixed with
            HGB by alpha.   "pool": probabilities mixed first (alpha), weighted once.
grid      posterior on D = pi * (mass moved to the nearest D that is a multiple of 30) + (1 - pi) * continuous.
snap      point prediction moved to the nearest value with t + pred a multiple of 30.
smooth    s_t = a * max(s_(t-1) - 5, 0) + (1 - a) * p_t (cv_utils.causal_smooth).

Selection (pi, a, alpha): nested LOFO over the 10 dual-CV folds (mix_calib.nested_select: loss summed over
the other 9 folds with the held-out fold's recordings masked). Scores: official binned_mae per recording
(~/projects/2026-competition/benchmark_utils/metrics.py). The public test (40 recordings, train+val model
~/neuralbench/save/mix3) is scored once at the end with the parameters chosen on all 10 folds.

Usage: python scripts/decision_layer.py   (writes OUT/decision_layer.json, OUT/declayer_<variant>_*_preds.npz)
"""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ablation_c import ABLATIONS, build_table_ext
from cv_utils import apply_bin_edges, causal_smooth
from dist_hgb import VARIANTS, class_stats, to_class
from dual_cv import OUT, load_trainval, seen_splits, unseen_splits
from ensemble_dist import GRID, MEMBERS, grid_omega, to_grid
from mix_calib import nested_select
from range_breakdown import load_preds
from stream_utils import EDGES, RANGE_NAMES, WEIGHTS, per_recording_scores, window_weights
from train_hgb import SAVE

_spec = importlib.util.spec_from_file_location("official_metrics",
                                               Path.home() / "projects/2026-competition/benchmark_utils/metrics.py")
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)

FEAT_SET = "dyn_maxmin_spec_eye"
HGB_MEMBERS = ["B", "B_h1", "B_h2", "B2"]
SEQ = ["uw_n10", "tcn_n10", "cnn3_emb"]
B = VARIANTS["B"]
REF = "mix3_gru_tcn_cnnemb"
ALPHA = 0.7
ALPHAS = np.round(np.arange(0.3, 1.01, 0.1), 2)
PIS = [0.0, 0.25, 0.5, 0.75, 0.86, 1.0]
AS = [0.0, 0.2, 0.4, 0.6, 0.8]
NF, NS = 600, 1201  # fine cells, slots
SLOT_Y = np.arange(NS) / 2.0  # atoms at integers (even slots), cell midpoints j + 0.5 (odd slots)
T_MAX = 301  # elapsed index t / 5 (train max 300)


# ---------------------------------------------------------------- exact metric weights
def range_counts(D):
    """D: array of onsets -> (n_windows per range (..., 4), S) for windows ending at 5, 10, ..., <= D."""
    D = np.asarray(D, dtype=np.float64)
    K = np.floor(D / 5.0 + 1e-9)
    c = [np.clip(np.floor((D - thr) / 5.0 + 1e-9), 0, K) for thr in (40.0, 90.0, 300.0)]
    n = np.stack([K - c[0], c[0] - c[1], c[1] - c[2], c[2]], axis=-1)
    S = (np.where(n > 0, WEIGHTS, 0.0)).sum(-1)
    return n, S


def w_exact(D, y):
    """Metric weight of a window with (clipped) target y in a recording with onset D (per recording)."""
    n, S = range_counts(D)
    b = np.clip(np.searchsorted(EDGES, np.minimum(y, 600.0), side="right") - 1, 0, 3)
    nb = np.take_along_axis(n, b[..., None], axis=-1)[..., 0]
    return WEIGHTS[b] / (np.maximum(nb, 1.0) * S)


def exact_table(D_train):
    """(T_MAX, NS) weights for t = 5 * index; slot 1200 = point class (expectation over D_train >= t + 600)."""
    t = 5.0 * np.arange(T_MAX)[:, None]
    tab = w_exact(t + SLOT_Y[None, :], np.broadcast_to(SLOT_Y[None, :], (T_MAX, NS)).copy())
    wD = w_exact(D_train, np.full(len(D_train), 600.0))
    for i in range(T_MAX):
        m = D_train >= 5.0 * i + 600.0
        tab[i, NS - 1] = wD[m].mean() if m.any() else tab[i, NS - 1]
    tab[0] = tab[1]  # t = 0 never occurs
    return tab


# ---------------------------------------------------------------- distributions on slots
def fine_matrix():
    """(25 grid cells) -> (600 unit cells + point class)."""
    A = np.zeros((len(GRID) - 1, NF + 1))
    for g in range(len(GRID) - 1):
        lo, hi = GRID[g], GRID[g + 1]
        if hi == lo:
            A[g, NF] = 1.0
        else:
            A[g, int(lo):int(hi)] = 1.0 / (hi - lo)
    return A


FINE = fine_matrix()


def atom_matrix(r):
    """t mod 30 = r -> (600 cells) x NS matrix moving each unit cell to its nearest on-grid atom."""
    y0 = (30 - r) % 30
    atoms = np.arange(y0, 600.0 + 1e-9, 30.0)
    G = np.zeros((NF, NS))
    mid = np.arange(NF) + 0.5
    a = atoms[np.abs(mid[:, None] - atoms[None, :]).argmin(1)]
    G[np.arange(NF), (2 * a).astype(int)] = 1.0
    return G


ATOM = {r: atom_matrix(r) for r in range(0, 30, 5)}


def to_slots(fine, t, pi):
    """fine (N, 601) probabilities, t (N,) window stops -> (N, NS) slot masses."""
    S = np.zeros((len(fine), NS))
    S[:, 1::2] = (1.0 - pi) * fine[:, :NF]
    if pi > 0:
        r = np.round(t).astype(int) % 30
        for rr in np.unique(r):
            m = r == rr
            S[m] += pi * (fine[m, :NF] @ ATOM[int(rr)])
    S[:, NS - 1] += fine[:, NF]
    return S


def slot_median(M):
    cum = np.cumsum(M, axis=1)
    cum /= cum[:, -1:]
    k = np.minimum((cum < 0.5).sum(axis=1), NS - 1)
    rows = np.arange(len(M))
    before = np.where(k > 0, cum[rows, np.maximum(k - 1, 0)], 0.0)
    frac = np.clip((0.5 - before) / np.maximum(cum[rows, k] - before, 1e-300), 0, 1)
    return np.where(k % 2 == 0, k / 2.0, (k - 1) / 2.0 + frac)


def old_slot_weights(omega_b, omega_grid):
    """Class weights of the submitted model on the slots (sequence models, HGB)."""
    y = np.minimum(SLOT_Y, 600.0)
    ws = omega_b[to_class(y, B)]
    wh = omega_grid[np.clip(np.searchsorted(GRID[:-1], y, side="right") - 1, 0, len(GRID) - 3)]
    wh[NS - 1] = omega_grid[-1]  # point cell
    return ws, wh


def predict(f, weights="old", pi=0.0, structure="comp", alpha=ALPHA, chunk=4096):
    out = np.empty(len(f["t"]))
    for i0 in range(0, len(out), chunk):
        sl = slice(i0, i0 + chunk)
        t = f["t"][sl]
        seq = [to_slots(x[sl], t, pi) for x in f["seq"]]
        hgb = to_slots(f["hgb"][sl], t, pi)
        if weights == "exact":
            ws = wh = f["wtab"][np.round(t / 5.0).astype(int)]
        else:
            ws, wh = f["w_old"]
        if structure == "comp":
            q = np.mean([(s * ws) / (s * ws).sum(1, keepdims=True) for s in seq], axis=0)
            h = hgb * wh
            M = alpha * q + (1 - alpha) * h / h.sum(1, keepdims=True)
        else:
            M = alpha * np.mean(seq, axis=0) * ws + (1 - alpha) * hgb * wh
        out[sl] = slot_median(M)
    return out


def snap(f, pred):
    t = f["t"]
    p = np.round((t + pred) / 30.0) * 30.0 - t
    p = np.where(p < 0, p + 30.0, p)
    return np.clip(p, 0.0, 600.0)


def smooth(f, pred, a):
    return causal_smooth(f["dte"], pred, a)


# ---------------------------------------------------------------- data
def rec_onsets(d):
    last = d.sort_values("stop").groupby("timeline").tail(1)
    return pd.Series((last.stop + last.target).to_numpy(), index=last.timeline.to_numpy())


def build_folds(d, protocols):
    hgb = {m: np.load(OUT / f"ens_{FEAT_SET}_{m}.npz", allow_pickle=True)["probas"].item() for m in HGB_MEMBERS}
    onset = rec_onsets(d)
    folds = []
    for proto, splits in protocols.items():
        zs = [np.load(OUT / f"gru_{n}_{proto}_proba.npz") for n in SEQ]
        for k, (tr, te) in enumerate(splits):
            dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
            _, ob = class_stats(dtr, B)
            D_tr = onset.loc[dtr.timeline.unique()].to_numpy()
            folds.append(dict(
                proto=proto, k=k, dte=dte, t=dte.stop.to_numpy().astype(np.float64),
                seq=[to_grid(z[f"arr_{k}"].astype(np.float64), B) @ FINE for z in zs],
                hgb=np.mean([to_grid(hgb[m][proto][k], MEMBERS[m][0]) for m in HGB_MEMBERS], axis=0) @ FINE,
                w_old=old_slot_weights(ob, grid_omega(dtr)), wtab=exact_table(D_tr),
                pi_train=float(np.mean(np.isclose(D_tr % 30, 0) | np.isclose(D_tr % 30, 30))),
                D=onset.loc[dte.timeline.to_numpy()].to_numpy()))
    return folds


# ---------------------------------------------------------------- scoring (official metric)
def official_rec_scores(dte, pred):
    f = lambda yt, yp: official.binned_mae(yt, yp, tuple(EDGES), tuple(WEIGHTS))
    codes, uniq = pd.factorize(dte.timeline)
    return pd.Series(official.group_scores(f, dte.target.to_numpy(), np.asarray(pred, float), codes), index=uniq)


def evaluate(folds, preds):
    """-> dict: per protocol fold scores, contributions, subgroup means (per-recording scores pooled over folds)."""
    res = {}
    for proto in ("unseen", "seen"):
        fs, contrib, recs = [], [], []
        for f, p in zip(folds, preds):
            if f["proto"] != proto:
                continue
            r = official_rec_scores(f["dte"], p)
            local = per_recording_scores(f["dte"], p)
            assert np.allclose(r.sort_index().to_numpy(), local.wbmae.sort_index().to_numpy(), atol=1e-9)
            fs.append(float(r.mean()))
            mae = local[RANGE_NAMES].to_numpy()
            w = np.where(np.isnan(mae), 0.0, WEIGHTS)
            contrib.append((np.nan_to_num(mae) * w / w.sum(1, keepdims=True)).mean(0))
            D = pd.Series(f["D"], index=f["dte"].timeline.to_numpy()).groupby(level=0).first().loc[r.index]
            recs.append(pd.DataFrame({"w": r.to_numpy(), "D": D.to_numpy()}))
        R = pd.concat(recs)
        on = np.isclose(R.D % 30, 0) | np.isclose(R.D % 30, 30)
        res[proto] = dict(mean=float(np.mean(fs)), folds=fs, contrib=np.mean(contrib, axis=0).tolist(),
                          short=float(R.w[R.D <= 330].mean()), on_grid=float(R.w[on].mean()),
                          off_grid=float(R.w[~on].mean()), n_short=int((R.D <= 330).sum()),
                          n_on=int(on.sum()), n_off=int((~on).sum()))
    res["macro"] = {k: ((np.array(res["unseen"][k]) + np.array(res["seen"][k])) / 2).tolist()
                    for k in ("mean", "contrib", "short", "on_grid", "off_grid")}
    return res


def by_proto(folds, preds):
    out = {"unseen": [], "seen": []}
    for f, p in zip(folds, preds):
        out[f["proto"]].append(np.asarray(p, np.float32))
    return out


def select(folds, cands):
    """cands: {param: list of per-fold preds}. Nested LOFO; also the all-fold choice (for the test set)."""
    keys = list(cands)
    stack = [np.stack([cands[k][i] for k in keys]) for i in range(len(folds))]
    preds, choices = nested_select(folds, stack, [np.zeros(len(f["dte"]), int) for f in folds], 1)
    tot = np.zeros(len(keys))
    for f, s in zip(folds, stack):
        tot += (np.abs(s - f["dte"].target.to_numpy()[None]) * window_weights(f["dte"])[None]).sum(1)
    return preds, [keys[c[0]] for c in choices], keys[int(tot.argmin())]


# ---------------------------------------------------------------- public test (train+val model, once)
def test_fold(model_dir="mix3"):
    """Batched component probabilities of a trained mix3 bundle (SAVE/<model_dir>) on the public test."""
    from cnn_encoder import RAW, Encoder, infer
    from gru_cv import ARCHS, CONFIGS, pad, seq_inputs
    b = joblib.load(SAVE / model_dir / "model.joblib")
    wg, sg, _ = ABLATIONS[FEAT_SET]
    df, X, _ = build_table_ext(wg, sg)
    mt = (df.split == "test").to_numpy()
    d_te, d_tv = df[mt].reset_index(drop=True), df[~mt].reset_index(drop=True)
    zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
    Fte = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)[mt]
    recs = [g.sort_values("stop").index.to_numpy() for _, g in d_te.groupby("timeline", sort=False)]
    cfg = CONFIGS["uw"]

    def seq_probs(arch, states, mu, sd, extra=None, **kw):
        out = []
        for i, sdict in enumerate(states):
            net = ARCHS[arch](len(mu), len(B) - 1, cfg["hidden"], cfg["dropout"], cfg["in_dropout"], **kw)
            net.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
            net.eval()
            e = None if extra is None else extra[i]
            xt = torch.tensor(pad([(seq_inputs(Fte[r], None if e is None else e[r]) - mu) / sd for r in recs]))
            with torch.no_grad():
                P = torch.softmax(net(xt)[0], -1).numpy()
            probs = np.empty((len(d_te), len(B) - 1))
            for j, r in enumerate(recs):
                probs[r] = P[j, :len(r)]
            out.append(probs)
        return np.mean(out, axis=0)

    Xraw = torch.as_tensor(np.asarray(np.load(RAW, mmap_mode="r")[mt]))
    embs = []
    for sdict in b["cnn_states"]:
        enc = Encoder()
        enc.load_state_dict({k: torch.tensor(v) for k, v in sdict.items()})
        embs.append(infer(enc.eval(), Xraw, torch.arange(len(d_te)))[1].numpy().astype(np.float32))
    Xh = X[mt]
    ex_g = ex_t = None
    if b.get("n2det_state") is not None:  # Work 2: N2-detector summaries of the test windows (full detector, as the wrapper)
        from n2det_stream import N2Summary
        det = Encoder(2, int(b.get("n2det_emb", 32)))
        det.load_state_dict({k: torch.tensor(v) for k, v in b["n2det_state"].items()})
        det.eval()
        with torch.no_grad():
            xr = torch.clamp(Xraw.float() / float(b["n2det_scale"]), -float(b["n2det_clip"]), float(b["n2det_clip"]))
            p = torch.cat([torch.softmax(det(xr[i:i + 2048])[0], -1)[:, 1] for i in range(0, len(xr), 2048)]).numpy()
        n2 = np.zeros((len(d_te), 5), np.float32)
        st = N2Summary(5.0)
        for r in recs:
            st.reset()
            for i in r:
                n2[i] = st.update(np.float64(p[i]))
        Xh = np.concatenate([Xh, n2], axis=1)
        into = tuple(b.get("n2det_into", ()))
        ex_g = [n2] * len(b["gru_states"]) if "gru" in into else None
        ex_t = [n2] * len(b["tcn_states"]) if "tcn" in into else None
        if "gemb" in into:
            embs = [np.concatenate([e, n2], axis=1) for e in embs]
    Pg = seq_probs("gru", b["gru_states"], b["gru_mu"], b["gru_sd"], extra=ex_g)
    Pt = seq_probs("tcn", b["tcn_states"], b["tcn_mu"], b["tcn_sd"], extra=ex_t)
    Pe = seq_probs("gru", b["gemb_states"], b["gemb_mu"], b["gemb_sd"], extra=embs)
    Xb = apply_bin_edges(Xh, b["bin_edges"])
    Ph = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(b["models"], b["member_edges"])], axis=0)
    D_tv = rec_onsets(d_tv).to_numpy()
    onset = rec_onsets(d_te)
    return dict(proto="test", k=0, dte=d_te, t=d_te.stop.to_numpy().astype(np.float64),
                seq=[to_grid(P, B) @ FINE for P in (Pg, Pt, Pe)], hgb=Ph @ FINE,
                w_old=old_slot_weights(np.asarray(b["gru_omega_b"]), np.asarray(b["grid_omega"])),
                wtab=exact_table(D_tv), pi_train=float(np.mean(np.isclose(D_tv % 30, 0) | np.isclose(D_tv % 30, 30))),
                D=onset.loc[d_te.timeline.to_numpy()].to_numpy())


# ---------------------------------------------------------------- checks
def check_weights(d):
    """w_exact against the official metric: perturb one prediction by 1 s -> metric change = window weight."""
    onset = rec_onsets(d)
    on = np.isclose(onset % 30, 0)
    picks = [onset[on].idxmin(), onset[on].idxmax(), onset[~on].index[0], onset[~on].index[-1],
             onset[(onset > 300) & (onset < 420) & on].index[0], onset[(onset > 600) & (onset < 700) & on].index[0]]
    worst = 0.0
    for tl in picks:
        g = d[d.timeline == tl].sort_values("stop")
        y, t = g.target.to_numpy(), g.stop.to_numpy()
        w_off = np.array([official.binned_mae(y, y + (np.arange(len(y)) == i), tuple(EDGES), tuple(WEIGHTS))
                          for i in range(len(y))])
        w_mine = w_exact(np.full(len(y), onset[tl]), y)
        w_loc = window_weights(g.reset_index(drop=True)) * 1.0  # one recording -> N_rec = 1
        diff = float(np.abs(w_off - w_mine).max())
        worst = max(worst, diff)
        print(f"  {tl.split(':')[1]:34s} onset {onset[tl]:8.2f} windows {len(y):3d}  max|official - w_exact| {diff:.2e}"
              f"  max|official - stream_utils| {float(np.abs(w_off - w_loc).max()):.2e}  sum w {w_off.sum():.6f}")
        tab = exact_table(np.array([onset[tl]]))  # table lookup (point class = this recording's D)
        w_tab = tab[np.round(t / 5).astype(int), np.where(y >= 600, NS - 1, np.round(2 * y).astype(int))]
        if np.isclose(onset[tl] % 5, 0):
            assert np.allclose(w_tab, w_off), "table lookup differs"
    print(f"  worst difference {worst:.2e}")
    return worst


if __name__ == "__main__":
    d, _, _ = load_trainval()
    protocols = {"unseen": unseen_splits(d), "seen": seen_splits(d)}
    print("== exact weight check against the official metric (6 real recordings)", flush=True)
    wdiff = check_weights(d)
    folds = build_folds(d, protocols)
    ref_saved = load_preds(REF, protocols)
    ref = [ref_saved[f["proto"]][f["k"]].astype(np.float64) for f in folds]
    eng = [predict(f) for f in folds]
    edev = max(float(np.abs(a - b).max()) for a, b in zip(eng, ref))
    print(f"== engine check: reference reproduced on slots, max |Δ| vs saved OOF = {edev:.2e} s", flush=True)
    print("   in-fold train grid ratio per fold:", [round(f["pi_train"], 3) for f in folds])

    V, sel = {}, {}  # variant -> per-fold preds ; selections

    def add(name, preds, choice=None):
        V[name] = preds
        if choice is not None:
            sel[name] = choice
        print("  done", name, "" if choice is None else choice, flush=True)

    add("ref", ref)
    # ---- experiment 1
    P1 = {s: [predict(f, "exact", 0.0, s) for f in folds] for s in ("comp", "pool")}
    add("E1 exact w, comp", P1["comp"])
    add("E1 exact w, pool", P1["pool"])
    add("E1x old w, pool", [predict(f, "old", 0.0, "pool") for f in folds])
    ca = {float(a): [predict(f, "exact", 0.0, "comp", a) for f in folds] for a in ALPHAS}
    p, ch, allc = select(folds, ca)
    add("E1 exact w, comp, alpha LOFO", p, dict(folds=ch, all=allc))
    # ---- experiment 2 (old weights)
    P2 = {pi: (ref if pi == 0 else [predict(f, "old", pi) for f in folds]) for pi in PIS}
    for pi in (0.5, 0.86, 1.0):
        add(f"E2 grid pi={pi}", P2[pi])
    add("E2 grid pi=train ratio", [predict(f, "old", f["pi_train"]) for f in folds])
    p, ch, allc = select(folds, P2)
    add("E2 grid pi LOFO", p, dict(folds=ch, all=allc))
    add("E2 snap(ref)", [snap(f, x) for f, x in zip(folds, ref)])
    # ---- experiment 3
    P3 = {a: [smooth(f, x, a) for f, x in zip(folds, ref)] for a in AS}
    for a in AS[1:]:
        add(f"E3 smooth a={a}", P3[a])
    p, ch, allc = select(folds, P3)
    add("E3 smooth a LOFO", p, dict(folds=ch, all=allc))
    # ---- experiment 4
    P12 = {pi: (P1["comp"] if pi == 0 else [predict(f, "exact", pi) for f in folds]) for pi in PIS}
    for pi in (0.5, 0.86, 1.0):
        add(f"E4 1+2 pi={pi}", P12[pi])
    p, ch, allc = select(folds, P12)
    add("E4 1+2 pi LOFO", p, dict(folds=ch, all=allc))
    add("E4 1+snap", [snap(f, x) for f, x in zip(folds, P1["comp"])])
    P13 = {a: [smooth(f, x, a) for f, x in zip(folds, P1["comp"])] for a in AS}
    p, ch, allc = select(folds, P13)
    add("E4 1+3 a LOFO", p, dict(folds=ch, all=allc))
    # 1+2+3, order A: grid in the distribution, then smoothing
    PA = {(pi, a): [smooth(f, x, a) for f, x in zip(folds, P12[pi])] for pi in PIS for a in AS}
    p, ch, allc = select(folds, PA)
    add("E4 1+2+3 grid->smooth, (pi,a) LOFO", p, dict(folds=ch, all=allc))
    # order B: smoothing, then snap to the grid
    PB = {a: [snap(f, x) for f, x in zip(folds, P13[a])] for a in AS}
    p, ch, allc = select(folds, PB)
    add("E4 1+3+snap smooth->snap, a LOFO", p, dict(folds=ch, all=allc))
    # order C: grid in the distribution, smoothing, snap again
    PC = {(pi, a): [snap(f, x) for f, x in zip(folds, PA[(pi, a)])] for pi in PIS for a in AS}
    p, ch, allc = select(folds, PC)
    add("E4 1+2+3 grid->smooth->snap, (pi,a) LOFO", p, dict(folds=ch, all=allc))
    # old weights: grid + smoothing (2+3), for completeness
    P23 = {(pi, a): [smooth(f, x, a) for f, x in zip(folds, P2[pi])] for pi in PIS for a in AS}
    p, ch, allc = select(folds, P23)
    add("E4x 2+3 grid->smooth, (pi,a) LOFO", p, dict(folds=ch, all=allc))

    # ---- report
    E = {k: evaluate(folds, v) for k, v in V.items()}
    r0 = E["ref"]
    print(f"\n{'variant':42s} unseen   seen  macro      Δ  better | contrib Δ [0,40) [40,90) [90,300) [300,600] |"
          f" onset<=330 (Δ) | on-grid Δ  off-grid Δ")
    rows = {}
    for k, e in E.items():
        du = np.array(e["unseen"]["folds"]) - np.array(r0["unseen"]["folds"])
        ds = np.array(e["seen"]["folds"]) - np.array(r0["seen"]["folds"])
        nb = int((du < -1e-9).sum() + (ds < -1e-9).sum())
        dc = np.array(e["macro"]["contrib"]) - np.array(r0["macro"]["contrib"])
        rows[k] = dict(unseen=e["unseen"]["mean"], seen=e["seen"]["mean"], macro=e["macro"]["mean"],
                       delta=e["macro"]["mean"] - r0["macro"]["mean"], better=nb, d_unseen=du.tolist(), d_seen=ds.tolist(),
                       contrib=e["macro"]["contrib"], d_contrib=dc.tolist(), short=e["macro"]["short"],
                       short_unseen=e["unseen"]["short"], short_seen=e["seen"]["short"],
                       on_grid=e["macro"]["on_grid"], off_grid=e["macro"]["off_grid"],
                       on_unseen=e["unseen"]["on_grid"], on_seen=e["seen"]["on_grid"],
                       off_unseen=e["unseen"]["off_grid"], off_seen=e["seen"]["off_grid"], sel=sel.get(k))
        print(f"{k:42s} {e['unseen']['mean']:6.2f} {e['seen']['mean']:6.2f} {e['macro']['mean']:6.2f} "
              f"{rows[k]['delta']:+6.2f}  {nb:2d}/10 | " + " ".join(f"{x:+8.2f}" for x in dc) +
              f" | {e['macro']['short']:6.2f} ({e['macro']['short'] - r0['macro']['short']:+.2f}) | "
              f"{e['macro']['on_grid'] - r0['macro']['on_grid']:+6.2f}  {e['macro']['off_grid'] - r0['macro']['off_grid']:+6.2f}")
    print(f"\nref: contrib {np.round(r0['macro']['contrib'], 2).tolist()}  onset<=330 {r0['macro']['short']:.2f} "
          f"(n unseen {r0['unseen']['n_short']}, seen {r0['seen']['n_short']})  on-grid {r0['macro']['on_grid']:.2f} "
          f"(n {r0['unseen']['n_on']}/{r0['seen']['n_on']})  off-grid {r0['macro']['off_grid']:.2f} "
          f"(n {r0['unseen']['n_off']}/{r0['seen']['n_off']})")
    for k, s in sel.items():
        print(f"  selection {k}: folds {s['folds']}  all-fold {s['all']}")
    for k, v in V.items():
        if k != "ref":
            P = by_proto(folds, v)
            tag = "".join(c if c.isalnum() else "_" for c in k)
            for pr in P:
                np.savez(OUT / f"declayer_{tag}_{pr}_preds.npz", *P[pr])

    # ---- public test, once, parameters = all-fold choices
    print("\n== public test (40 recordings, train+val model, scored once)", flush=True)
    ft = test_fold()
    saved = np.load(SAVE / "mix3" / "test_preds.npz", allow_pickle=True)
    assert (saved["timeline"] == ft["dte"].timeline.to_numpy()).all()
    tref = predict(ft)
    print(f"  reference reproduced: max |Δ| vs saved test preds {float(np.abs(tref - saved['mix3_a0.7']).max()):.2e} s")
    T = {"ref": tref}
    e1 = predict(ft, "exact")
    T["E1 exact w, comp"] = e1
    T["E1 exact w, pool"] = predict(ft, "exact", 0.0, "pool")
    T["E2 grid pi LOFO"] = predict(ft, "old", sel["E2 grid pi LOFO"]["all"])
    T["E2 grid pi=0.86"] = predict(ft, "old", 0.86)
    T["E2 snap(ref)"] = snap(ft, tref)
    T["E3 smooth a LOFO"] = smooth(ft, tref, sel["E3 smooth a LOFO"]["all"])
    pi12 = sel["E4 1+2 pi LOFO"]["all"]
    T["E4 1+2 pi LOFO"] = predict(ft, "exact", pi12)
    T["E4 1+snap"] = snap(ft, e1)
    T["E4 1+3 a LOFO"] = smooth(ft, e1, sel["E4 1+3 a LOFO"]["all"])
    pi, a = sel["E4 1+2+3 grid->smooth, (pi,a) LOFO"]["all"]
    T["E4 1+2+3 grid->smooth, (pi,a) LOFO"] = smooth(ft, predict(ft, "exact", pi), a)
    a = sel["E4 1+3+snap smooth->snap, a LOFO"]["all"]
    T["E4 1+3+snap smooth->snap, a LOFO"] = snap(ft, smooth(ft, e1, a))
    pi, a = sel["E4 1+2+3 grid->smooth->snap, (pi,a) LOFO"]["all"]
    T["E4 1+2+3 grid->smooth->snap, (pi,a) LOFO"] = snap(ft, smooth(ft, predict(ft, "exact", pi), a))
    test = {}
    on_t = np.isclose(ft["D"] % 30, 0) | np.isclose(ft["D"] % 30, 30)
    print(f"  test recordings on grid: {int(pd.Series(on_t, index=ft['dte'].timeline).groupby(level=0).first().sum())}/40")
    for k, p in T.items():
        test[k] = float(official_rec_scores(ft["dte"], p).mean())
        print(f"  {k:42s} {test[k]:.4f}  ({test[k] - test['ref']:+.2f})")
    (OUT / "decision_layer.json").write_text(json.dumps(dict(
        weight_check_max_diff=wdiff, engine_max_diff=edev, rows=rows, test=test,
        ref=dict(contrib=r0["macro"]["contrib"], short=r0["macro"]["short"], on_grid=r0["macro"]["on_grid"],
                 off_grid=r0["macro"]["off_grid"], n={p: {k: r0[p][k] for k in ("n_short", "n_on", "n_off")}
                                                      for p in ("unseen", "seen")})), indent=2, default=str))
