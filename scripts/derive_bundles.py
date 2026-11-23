"""Reproducibility (23 Nov): the three bundle derivations of the best sealed submission's chain that were done inline at the time
(18-20 Nov), as code. Each derived bundle is written to SAVE/<name>_rederived and compared with the bundle that was actually packaged
(every key; HGB models by every raw node field, baseline and exported arrays).

  ctx_c3_all       <- ctx_c2_all:      drop the TCN and GRU-emb keys, seq_comps ("gru",), mix_alpha 0.5 -> 0.4
  ctxE_c2hs3p_all  <- ctxE_c2hs3_all:  HGB E models B s0-2, B_h1 s0, B_h2 s0-2, B2 s0 (of members x seeds 0-2), member_weights
                                       1/12 x3, 1/4, 1/12 x3, 1/4 (every member 1/4)
  mixI_all         <- ctxE_c2hs3p_all + ctx_c3_all: second HGB set = ctx_c3's 4 HGB ctx models with their own bin edges and grid
                                       omega, hgb2_weight 0.5, seq_weights [2/3, 1/6, 1/6] (= 0.5 ctx_c3_540 + 0.5 ctxE_c2hs3p_540)

Usage: python scripts/derive_bundles.py   (prints one line per bundle: IDENTICAL / DIFFERS + the differing keys)
"""
import json

import joblib
import numpy as np

from train_hgb import SAVE
from tree_numpy import export_hgb

TCN_GEMB = ("tcn_states", "tcn_mu", "tcn_sd", "tcn_dilations", "cnn_states", "cnn_scale", "cnn_clip", "cnn_emb", "gemb_states",
            "gemb_mu", "gemb_sd")
C2HS3P_IDX = [0, 1, 2, 3, 6, 7, 8, 9]  # of B_s0-2, B_h1_s0-2, B_h2_s0-2, B2_s0-2


def is_hgb(x):
    return hasattr(x, "_predictors")


def same_hgb(a, b):
    pa = [p.nodes for it in a._predictors for p in it]
    pb = [p.nodes for it in b._predictors for p in it]
    if len(pa) != len(pb) or any(x.dtype != y.dtype or x.tobytes() != y.tobytes() for x, y in zip(pa, pb)):
        return False
    if np.asarray(a._baseline_prediction).tobytes() != np.asarray(b._baseline_prediction).tobytes():
        return False
    ea, eb = export_hgb(a), export_hgb(b)
    return all(np.asarray(ea[k]).tobytes() == np.asarray(eb[k]).tobytes() for k in ea)


def same(a, b):
    if is_hgb(a) or is_hgb(b):
        return is_hgb(a) and is_hgb(b) and same_hgb(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes() == b.tobytes())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def load(name):
    return joblib.load(SAVE / name / "model.joblib")


def derive_ctx_c3(c2):
    b = {k: v for k, v in c2.items() if k not in TCN_GEMB}
    b["mix_alpha"] = 0.4
    b["seq_comps"] = ("gru",)
    return b


def derive_c2hs3p(c2hs3):
    b = dict(c2hs3)
    b["models"] = [c2hs3["models"][i] for i in C2HS3P_IDX]
    b["member_edges"] = [c2hs3["member_edges"][i] for i in C2HS3P_IDX]
    b["member_weights"] = [1 / 12] * 3 + [0.25] + [1 / 12] * 3 + [0.25]
    return b


def derive_mixI(c2hs3p, c3):
    b = dict(c2hs3p)
    b.update(hgb2_models=c3["models"], hgb2_member_edges=c3["member_edges"], hgb2_bin_edges=c3["bin_edges"],
             hgb2_grid_omega=c3["grid_omega"], hgb2_member_weights=None, hgb2_weight=0.5, seq_weights=[2 / 3, 1 / 6, 1 / 6])
    return b


if __name__ == "__main__":
    c3 = derive_ctx_c3(load("ctx_c2_all"))
    c2hs3p = derive_c2hs3p(load("ctxE_c2hs3_all"))
    mixI = derive_mixI(c2hs3p, c3)
    report = {}
    for name, b in (("ctx_c3_all", c3), ("ctxE_c2hs3p_all", c2hs3p), ("mixI_all", mixI)):
        ref = load(name)
        bad = sorted(set(ref) ^ set(b)) + [k for k in set(ref) & set(b) if not same(ref[k], b[k])]
        out = SAVE / f"{name}_rederived"
        out.mkdir(exist_ok=True)
        joblib.dump(b, out / "model.joblib")
        report[name] = "IDENTICAL" if not bad else f"DIFFERS: {bad}"
        print(f"{name}: {report[name]}  ({len(b)} keys) -> {out}", flush=True)
    (SAVE / "derive_bundles.json").write_text(json.dumps(report, indent=2))
