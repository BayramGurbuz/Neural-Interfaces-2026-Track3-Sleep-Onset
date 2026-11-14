"""Diagnosis of the stream-start sensitivity (which parts of the inputs carry the "recording start" signature).
Train+val mix3 bundle on the 40 test recordings (stream_start_test); for k = 6 and 24 the truncated-stream inputs get
one group of columns at a time replaced by their full-history values (as if the model had seen the dropped windows for
that group only): HGB-only score with each of the 817-feature groups patched (raw, rel = diff to the first-2-min mean,
EMAs, EMA rel, EMA minus running extrema, slopes / var / ac1, 10-min EMA, elapsed) and cumulatively; sequence-only
score with the first-2-min baseline columns, the elapsed column, or both patched (GRU / TCN / GRU-emb inputs).
Usage: python scripts/stream_start_diag.py   -> OUT/stream_start_diag.json
"""
import json
import re

import numpy as np
import torch

import stream_start_test as T
from gru_cv import seq_inputs

names = list(T.build_table_ext(T.wg, T.sg)[2])
GROUPS = {"elapsed": r"^elapsed_s$", "raw": r"^(?!elapsed)(?!.*_(ema|rel|slope\d+s|var300s|ac1_300s|minus_run)).*$",
          "rel (first-2-min diff)": r"_rel$", "ema30/120/300 (no rel)": r"_ema(30|120|300)s$",
          "ema minus runmax/min": r"_minus_run(max|min)$", "slopes/var300/ac1": r"_(slope\d+s|var300s|ac1_300s)$",
          "ema600": r"_ema600s$"}
gidx = {}
for g, pat in GROUPS.items():
    gidx[g] = [j for j, n in enumerate(names) if re.search(pat, n) and not (g == "raw" and n == "elapsed_s")]
# "_rel" also matches ema*_rel; keep groups disjoint: ema*_rel go to rel group (first-2-min difference of EMAs)
cover = sorted(sum(gidx.values(), []))
assert cover == list(range(817)), (len(cover), sorted(set(range(817)) - set(cover))[:10])


def hgb_pred(H):
    Xb = T.apply_bin_edges(H, T.b["bin_edges"])
    Ph = np.mean([T.to_grid(m.predict_proba(Xb), e) for m, e in zip(T.b["models"], T.b["member_edges"])], axis=0)
    Wh = Ph * np.asarray(T.b["grid_omega"])
    Wh /= Wh.sum(1, keepdims=True)
    return T.decide(Wh, None, np.ones(Wh.shape[1]), True, T.GRID)


def seq_pred(Fw, rows, xg, xe):
    with torch.no_grad():
        qg = T.wgrid(T.seq_mean(T.grus, (xg - T.b["gru_mu"]) / T.b["gru_sd"]))
        qt = T.wgrid(T.seq_mean(T.tcns, (xg - T.b["tcn_mu"]) / T.b["tcn_sd"]))
        qe = T.wgrid(np.mean([T.seq_mean([g], (x - T.b["gemb_mu"]) / T.b["gemb_sd"]) for g, x in zip(T.gembs, xe)], axis=0))
    qs = (qg + qt + qe) / 3
    return T.decide(qs, None, np.ones(qs.shape[1]), True, T.GRID)


if __name__ == "__main__":
    res = {}
    full_H = {}
    for r in T.recs:
        T.stream.reset()
        full_H[r[0]] = np.stack([T.stream.update(fi.astype(np.float64)) for fi in T.F[r]]).astype(np.float32)
    BASE = slice(T.F.shape[1], 2 * T.F.shape[1])
    for k0 in (6, 24):
        k = k0
        kk = [min(k0, len(r) - 1) for r in T.recs]  # a test recording shorter than k keeps its last window
        rows_list = [r[q:] for r, q in zip(T.recs, kk)]
        out = {}
        # HGB: truncated / full / per-group patched / cumulative
        Ht, Hf = [], []
        for r, q in zip(T.recs, kk):
            T.stream.reset()
            Ht.append(np.stack([T.stream.update(fi.astype(np.float64)) for fi in T.F[r[q:]]]).astype(np.float32))
            Hf.append(full_H[r[0]][q:])
        s_t = T.score(rows_list, [hgb_pred(H) for H in Ht])[0]
        s_f = T.score(rows_list, [hgb_pred(H) for H in Hf])[0]
        out["hgb"] = dict(truncated=s_t, full=s_f, delta=s_t - s_f, patched={}, cumulative={})
        Hc = [H.copy() for H in Ht]
        for g, idx in GROUPS.items():
            idx = gidx[g]
            Hp = []
            for H, Hfull in zip(Ht, Hf):
                Hx = H.copy()
                Hx[:, idx] = Hfull[:, idx]
                Hp.append(Hx)
            sp = T.score(rows_list, [hgb_pred(H) for H in Hp])[0]
            out["hgb"]["patched"][g] = dict(score=sp, recovered=s_t - sp, n_cols=len(idx))
            for H, Hfull in zip(Hc, Hf):
                H[:, idx] = Hfull[:, idx]
            out["hgb"]["cumulative"][g] = T.score(rows_list, [hgb_pred(H) for H in Hc])[0]
        # sequence: baseline columns / elapsed / both patched
        def seq_run(patch):
            preds = []
            for r, q in zip(T.recs, kk):
                rows = r[q:]
                Fw = T.F[rows]
                xg = seq_inputs(Fw)
                xg_full = seq_inputs(T.F[r])[q:]
                with torch.no_grad():
                    embs = [T.infer(e, torch.tensor(T.raw[rows]), torch.arange(len(rows)))[1].numpy().astype(np.float32) for e in T.encs]
                    embs_full = [T.infer(e, torch.tensor(T.raw[r]), torch.arange(len(r)))[1].numpy().astype(np.float32) for e in T.encs]
                xe = [seq_inputs(Fw, E) for E in embs]
                xe_full = [seq_inputs(T.F[r], E)[q:] for E in embs_full]
                if "base" in patch:
                    xg[:, BASE] = xg_full[:, BASE]
                    for x, xf in zip(xe, xe_full):
                        x[:, BASE] = xf[:, BASE]
                        x[:, T.EL_S + 1 + T.EMB_N:] = xf[:, T.EL_S + 1 + T.EMB_N:]  # embedding minus its first-2-min mean
                if "elapsed" in patch:
                    xg[:, T.EL_S] = xg_full[:, T.EL_S]
                    for x, xf in zip(xe, xe_full):
                        x[:, T.EL_S] = xf[:, T.EL_S]
                preds.append(seq_pred(Fw, rows, xg, xe))
            return T.score(rows_list, preds)[0]
        out["seq"] = {p: seq_run(p) for p in ("none", "elapsed", "base", "base+elapsed")}
        res[f"k{k0}"] = out
        print(f"k = {k0}: HGB-only truncated {s_t:.2f} full {s_f:.2f} Δ {s_t - s_f:+.2f}; recovered by patching one group: "
              + ", ".join(f"{g} {v['recovered']:+.2f}" for g, v in out["hgb"]["patched"].items()), flush=True)
        print(f"        cumulative (elapsed, +raw, +rel, +ema, +minus_run, +slopes, +ema600): "
              + " ".join(f"{v:.2f}" for v in out["hgb"]["cumulative"].values()), flush=True)
        print(f"        seq-only truncated {out['seq']['none']:.2f}, elapsed patched {out['seq']['elapsed']:.2f}, baseline patched "
              f"{out['seq']['base']:.2f}, both {out['seq']['base+elapsed']:.2f}", flush=True)
    (T.OUT / "stream_start_diag.json").write_text(json.dumps(res, indent=2))
