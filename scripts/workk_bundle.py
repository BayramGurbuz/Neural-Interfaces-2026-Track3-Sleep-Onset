"""Work K / Work J bundles (20 Nov), built from the Work I bundle SAVE/mixI_all (= 0.5 ctx_c3_540 + 0.5 ctxE_c2hs3p_540, sealed 32.92).
No new components, no retraining of existing ones; the recipe (alpha 0.4, HGB 1/2 HGB ctx + 1/2 HGB E-s3p, sequence 2/3 GRU ctx + 1/6 TCN +
1/6 GRU-emb) stays, only seed averages change:

  K   : GRU ctx = 20 seeds (0-9 = gru_ctx_all, 10-19 = gru_ctx_k_all), as two groups of 10 with weights 1/2, 1/2 (= the mean of 20);
        HGB ctx members B and B_h2 x random_state 0, 1, 2 (ens_c_all_ctx_s3p_crop; same table / bins / n_iter), every member 1/4
        (seed models 1/12 each), as in s3p.
  K3  : K + HGB E members B_h1 and B2 x random_state 0, 1, 2 too (ens_c_all_ctxE_s3_crop, Work G): all 12 HGB E models 1/12.
  K0, K30: K, K3 with the added seeds' weights 0 (GRU groups 1, 0; added HGB models 0, the others mixI's exact weights) -> must stream
        bit-identically to mixI_540.
  J   : Work J = (ctx_c3_540 + ctx_c2_540 + ctxE_c2hs3p_540) / 3 with the mixI components: alpha 13/30, sequence weights 7 : 3 : 3
        (GRU ctx, TCN, GRU-emb), HGB ctx share 11/17 (ctx_c2 = alpha 0.5, (GRU+TCN+GRU-emb)/3, HGB ctx; its HGB ctx, GRU ctx, TCN and
        GRU-emb are the files of the other two packages).

Checks here: seed-0 models of the new trainings are bit-identical to the existing ones (GRU state dicts, mu / sd; exported HGB tree arrays,
bin edges), the reused HGB E models of mixI are the Work G s3 models.
Usage: python scripts/workk_bundle.py K|K3|K0|K30|J  -> SAVE/mix<variant>_all
"""
import json
import sys
from fractions import Fraction

import joblib
import numpy as np

from train_hgb import SAVE
from tree_numpy import export_hgb

V = sys.argv[1]
assert V in ("K", "K3", "K0", "K30", "J")


def same_tree(a, b):
    """Bit-identical fitted HGB models: every predictor's raw node array (all fields: thresholds, bin thresholds, values, gains,
    counts, structure), the baseline, and the exported arrays the wrapper ships."""
    pa = [p.nodes for it in a._predictors for p in it]
    pb = [p.nodes for it in b._predictors for p in it]
    if len(pa) != len(pb) or any(x.dtype != y.dtype or x.tobytes() != y.tobytes() for x, y in zip(pa, pb)):
        return False
    if np.asarray(a._baseline_prediction).tobytes() != np.asarray(b._baseline_prediction).tobytes():
        return False
    ea, eb = export_hgb(a), export_hgb(b)
    return all(np.asarray(ea[k]).tobytes() == np.asarray(eb[k]).tobytes() for k in ea)


def same_edges(a, b):
    return len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


b = joblib.load(SAVE / "mixI_all" / "model.joblib")
info = dict(variant=V, base="mixI_all")

if V == "J":
    b["mix_alpha"] = float(Fraction(13, 30))
    b["seq_weights"] = [7 / 13, 3 / 13, 3 / 13]
    b["hgb2_weight"] = float(Fraction(11, 17))
    info.update(recipe="(ctx_c3_540 + ctx_c2_540 + ctxE_c2hs3p_540) / 3", mix_alpha=b["mix_alpha"], seq_weights=b["seq_weights"],
                hgb2_weight=b["hgb2_weight"])
else:
    zero = V in ("K0", "K30")
    # ---- GRU ctx seeds 10-19
    g0, gk = joblib.load(SAVE / "gru_ctx_all" / "model.joblib"), joblib.load(SAVE / "gru_ctx_k_all" / "model.joblib")
    rk = json.loads((SAVE / "gru_ctx_k_all" / "results.json").read_text())
    assert rk["seeds"] == [0] + list(range(10, 20)) and rk["epochs"] == g0["epochs"] == 20, rk
    for k in ("gru_mu", "gru_sd", "gru_omega_b"):
        assert np.array_equal(gk[k], g0[k]) and np.array_equal(gk[k], b[k]), k
    s0n, s0 = gk["gru_states"][0], g0["gru_states"][0]
    assert s0n.keys() == s0.keys() and all(np.array_equal(s0n[k], s0[k]) for k in s0), "GRU seed 0 retrain differs"
    assert all(all(np.array_equal(x[k], y[k]) for k in x) for x, y in zip(b["gru_states"], g0["gru_states"])) and len(b["gru_states"]) == 10
    b["gru_states"] = list(b["gru_states"]) + list(gk["gru_states"][1:])
    b["gru_group_sizes"] = [10, 10]
    b["gru_group_weights"] = [1.0, 0.0] if zero else [0.5, 0.5]
    info.update(gru="seeds 0-9 gru_ctx_all + 10-19 gru_ctx_k_all", gru_seed0_bit_identical=True, gru_group_weights=b["gru_group_weights"])
    # ---- HGB ctx: B, B_h2 x seeds 0, 1, 2
    c0, cs = joblib.load(SAVE / "ens_c_all_ctx_crop" / "model.joblib"), joblib.load(SAVE / "ens_c_all_ctx_s3p_crop" / "model.joblib")
    rs = json.loads((SAVE / "ens_c_all_ctx_s3p_crop" / "results.json").read_text())
    assert list(rs["members"]) == ["B_s0", "B_s1", "B_s2", "B_h2_s0", "B_h2_s1", "B_h2_s2"], list(rs["members"])
    assert same_edges(cs["bin_edges"], c0["bin_edges"]) and same_edges(b["hgb2_bin_edges"], c0["bin_edges"])
    assert np.array_equal(cs["grid_omega"], c0["grid_omega"]) and np.array_equal(b["hgb2_grid_omega"], c0["grid_omega"])
    assert all(same_tree(x, y) for x, y in zip(b["hgb2_models"], c0["models"])) and len(b["hgb2_models"]) == 4
    assert same_tree(cs["models"][0], c0["models"][0]), "HGB ctx B seed 0 retrain differs"
    assert same_tree(cs["models"][3], c0["models"][2]), "HGB ctx B_h2 seed 0 retrain differs"
    ed = [np.asarray(e) for e in c0["member_edges"]]
    assert all(np.array_equal(np.asarray(cs["member_edges"][i]), ed[j]) for i, j in ((0, 0), (1, 0), (2, 0), (3, 2), (4, 2), (5, 2)))
    m = c0["models"]
    b["hgb2_models"] = [m[0], cs["models"][1], cs["models"][2], m[1], m[2], cs["models"][4], cs["models"][5], m[3]]
    b["hgb2_member_edges"] = [ed[0]] * 3 + [ed[1]] + [ed[2]] * 3 + [ed[3]]
    b["hgb2_member_weights"] = ([0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25] if zero else [1 / 12] * 3 + [0.25] + [1 / 12] * 3 + [0.25])
    info.update(hgb2="ctx: B s0-2, B_h1 s0, B_h2 s0-2, B2 s0", hgb_ctx_seed0_bit_identical=True, hgb2_member_weights=b["hgb2_member_weights"])
    if V in ("K3", "K30"):
        e3 = joblib.load(SAVE / "ens_c_all_ctxE_s3_crop" / "model.joblib")
        r3 = json.loads((SAVE / "ens_c_all_ctxE_s3_crop" / "results.json").read_text())
        names = list(r3["members"])
        assert names == ["B_s0", "B_s1", "B_s2", "B_h1_s0", "B_h1_s1", "B_h1_s2", "B_h2_s0", "B_h2_s1", "B_h2_s2", "B2_s0", "B2_s1", "B2_s2"]
        assert same_edges(e3["bin_edges"], b["bin_edges"]) and np.array_equal(e3["grid_omega"], b["grid_omega"])
        # mixI's 8 HGB E models = s3 models B_s0-2, B_h1_s0, B_h2_s0-2, B2_s0 (ctxE_c2hs3p)
        idx = [0, 1, 2, 3, 6, 7, 8, 9]
        assert all(same_tree(b["models"][i], e3["models"][j]) for i, j in enumerate(idx))
        wI = list(b["member_weights"])
        b["models"] = list(e3["models"])
        b["member_edges"] = [np.asarray(e) for e in e3["member_edges"]]
        if zero:  # mixI's exact weights on its 8 models, 0 on the added 4
            w = [0.0] * 12
            for i, j in enumerate(idx):
                w[j] = wI[i]
            b["member_weights"] = w
        else:
            b["member_weights"] = [1 / 12] * 12
        info.update(hgb1="ctxE s3: all 12 models", member_weights=b["member_weights"])

out = SAVE / f"mix{V}_all"
out.mkdir(exist_ok=True)
joblib.dump(b, out / "model.joblib")
(out / "results.json").write_text(json.dumps(info, indent=2, default=float))
print("saved", out, json.dumps(info, default=float))
