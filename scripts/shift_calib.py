"""Work A (16 Nov): stream-start shift calibration of the submitted models, no retraining.

Each submitted model is replaced by its train+val (500 recordings) counterpart and run on the 40 public-test recordings
(out-of-sample, seen subjects only) with the stream started at window k: elapsed time, every causal feature (EMAs, dynamics,
running extrema, first-2-min baselines), the detector summaries and the recurrent / TCN states start there; targets are
unchanged, only the windows >= k are scored (official per-recording W-bMAE, binned_mae). Window-level quantities that do not
depend on the stream start (62 window features, CNN embeddings, detector outputs per window) are computed once.

Models (name: bundle in SAVE, decision):
  final540  mix_uw_n10   GRU-uw + HGB(c), alpha 0.6
  mix3_540  mix3         (GRU + TCN + GRU-emb)/3 + HGB(c), alpha 0.7
  gridpi    mix3         same + the 30 s onset-grid decision pi(D) (the submitted wrapper's own _grid_decide; elapsed counted
                         from the stream start, as on the server)
  cand1     cand1_tv     HGB(c)+n2det + GRU crop, alpha 0.6 (binary detector n2det_final_tv/p_full.npy)
  cand2     cand2_tv     HGB(c)+closeness+crop + GRU crop+closeness, alpha 0.6 (multi-class detector n2det_final_tv/close_full.npy)
Output per model: S[rec, k] for k = 0..K_MAX (cap: at least one window left), and S_rand[rec, j] for 3 random starts per
recording (uniform in [0, (onset - 60 s) / 5], seeds 0, 1, 2), k_rand[rec, j].

Usage: python scripts/shift_calib.py MODEL [MODEL ...] [--alpha-MODEL=A]   -> OUT/shift_calib_<MODEL>.npz (resumable per model)
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import torch

from ablation_c import ABLATIONS, build_table_ext
from causal_features import CausalFeatureStream, core_index
from cnn_encoder import Encoder, infer
from cv_utils import apply_bin_edges
from decision_layer import official_rec_scores
from dist_hgb import decide
from dual_cv import OUT
from ensemble_dist import GRID, to_grid
from gru_cv import ARCHS, CONFIGS, seq_inputs
from n2det_feats import summaries
from train_hgb import SAVE

torch.set_num_threads(2)
K_MAX = 60
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"final540": dict(bundle="mix_uw_n10"), "mix3_540": dict(bundle="mix3"),
          "gridpi": dict(bundle="mix3", grid=dict(cuts="900,1100,1300", values="1.0,0.592,0.644,0.444")),
          "cand1": dict(bundle="cand1_tv", n2="p_full.npy"), "cand2": dict(bundle="cand2_tv", n2="close_full.npy"),
          # Work B rows (HGB new + GRU crop+closeness fixed; alpha set with --alpha-<name>=A from the shifted-CV choice)
          "wb2": dict(bundle="cand2_tv", n2="close_full.npy", tcn_from="tcn_n2c_tv"),
          "wb3": dict(bundle="cand2_tv", n2="close_full.npy", tcn_from="tcn_n2c_tv", gemb_from="mix3crop"),
          "wb4": dict(bundle="cand2_tv", n2="close_full.npy", tcn_from="tcn_n2c_tv", comps=("gru", "tcn")),
          "wb5": dict(bundle="cand2_tv", n2="close_full.npy", comps=("gru",)),
          # alpha checks (17 Nov): candidate 2 at the Work B rows' alpha 0.5, row 5 at candidate 2's alpha 0.6
          "cand2a5": dict(bundle="cand2_tv", n2="close_full.npy"), "wb5a6": dict(bundle="cand2_tv", n2="close_full.npy", comps=("gru",))}

wg, sg, _ = ABLATIONS["dyn_maxmin_spec_eye"]
df, X817, _ = build_table_ext(wg, sg)
mt = (df.split == "test").to_numpy()
d = df[mt].reset_index(drop=True)
zb, ze = np.load(SAVE / "features_base.npz"), np.load(SAVE / "features_ext.npz")
F = np.concatenate([zb["feats"], ze["feats"]], axis=1).astype(np.float32)[mt]
raw = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
recs = [g.sort_values("stop").index.to_numpy() for _, g in d.groupby("timeline", sort=False)]
target = d.target.to_numpy().astype(np.float64)
onset = [float(d.stop[r[-1]] + target[r[-1]]) for r in recs]
X817 = X817[mt]


def grid_module(bundle_name, cuts, values):
    """Build the gridpi wrapper from the train+val bundle into the scratch output and import it (benchmark_utils stubbed)."""
    import subprocess
    out_root = OUT / "shift_calib_wrappers"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_hgb_submission.py"), "--model", str(SAVE / bundle_name / "model.joblib"),
                    "--name", "gridpi_tv", "--grid-pi-cuts", cuts, "--grid-pi-values", values, "--out-root", str(out_root), "--no-zip"],
                   check=True, capture_output=True)
    stub = types.ModuleType("benchmark_utils.base_solver")
    stub.CompetSolver = type("CompetSolver", (), {})
    sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
    sys.modules["benchmark_utils.base_solver"] = stub
    spec = importlib.util.spec_from_file_location("gridpi_tv", out_root / "gridpi_tv" / "submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    b = joblib.load(out_root / "gridpi_tv" / "model.joblib")
    return sub.HGBStreamModel(b.get("model"), 128.0, b.get("bin_edges"), b, [str(c) for c in zb["ch_names"]])


class Runner:
    def __init__(self, spec):
        b = joblib.load(SAVE / spec["bundle"] / "model.joblib")
        n2_into = tuple(b.get("n2det_into", ()))
        if spec.get("tcn_from"):
            bt = joblib.load(SAVE / spec["tcn_from"] / "model.joblib")
            b.update({k: bt[k] for k in ("tcn_states", "tcn_mu", "tcn_sd")})
            n2_into += ("tcn",) if bt.get("tcn_n2c") else ()
        if spec.get("gemb_from"):
            bg = joblib.load(SAVE / spec["gemb_from"] / "model.joblib")
            b.update({k: bg[k] for k in ("cnn_states", "gemb_states", "gemb_mu", "gemb_sd")})
        self.b, self.kind, self.alpha = b, b["kind"], float(spec.get("alpha", b["mix_alpha"]))
        self.comps = tuple(spec.get("comps", ("gru", "tcn", "gemb") if b["kind"] == "mix3" else ("gru",)))
        self.edges, self.omega_b = np.asarray(b["gru_edges"], np.float64), np.asarray(b["gru_omega_b"], np.float64)
        self.grid_omega = np.asarray(b["grid_omega"], np.float64)
        self.stream = CausalFeatureStream(5.0, tuple(b["groups"]), core_index(range(4)))
        cfg = CONFIGS["uw"]
        n_out = len(self.edges) - 1

        def nets(arch, states, n_in):
            out = []
            for s in states:
                net = ARCHS[arch](n_in, n_out, cfg["hidden"], cfg["dropout"], cfg["in_dropout"])
                net.load_state_dict({k: torch.tensor(v) for k, v in s.items()})
                out.append(net.eval())
            return out

        self.grus = nets("gru", b["gru_states"], len(b["gru_mu"]))
        self.n2_into = n2_into
        self.hgb_n2 = b.get("n2det_state") is not None or b.get("n2det_states") is not None
        self.n2v = np.load(SAVE / "n2det_final_tv" / spec["n2"]).astype(np.float64)[mt] if spec.get("n2") else None
        assert (self.n2v is not None) == (self.hgb_n2 or "gru" in self.n2_into or "tcn" in self.n2_into)
        if self.kind == "mix3":
            self.tcns = nets("tcn", b["tcn_states"], len(b["tcn_mu"]))
            self.gembs = nets("gru", b["gemb_states"], len(b["gemb_mu"]))
            embs = []
            for s in b["cnn_states"]:
                e = Encoder()
                e.load_state_dict({k: torch.tensor(v) for k, v in s.items()})
                with torch.no_grad():
                    embs.append(infer(e.eval(), torch.tensor(raw), torch.arange(len(raw)))[1].numpy().astype(np.float32))
            self.embs = embs
        self.gridm = grid_module(spec["bundle"], **spec["grid"]) if spec.get("grid") else None

    @staticmethod
    def seq_mean(models, x):
        with torch.no_grad():
            xt = torch.tensor(x)[None]
            return np.mean([torch.softmax(m(xt)[0][0], -1).numpy().astype(np.float64) for m in models], axis=0)

    def wq(self, P):
        q = P * self.omega_b
        return to_grid(q / q.sum(1, keepdims=True), self.edges)

    def predict(self, r, k):
        b = self.b
        rows = r[k:]
        Fw = F[rows]
        self.stream.reset()
        H = np.stack([self.stream.update(fi.astype(np.float64)) for fi in Fw]).astype(np.float32)
        if k == 0:
            assert np.abs(H - X817[rows]).max() == 0.0
        S2 = summaries(self.n2v[rows], [np.arange(len(rows))]).astype(np.float32) if self.n2v is not None else None
        Hx = np.concatenate([H, S2], axis=1) if self.hgb_n2 else H
        Xb = apply_bin_edges(Hx, b["bin_edges"])
        Ph = np.mean([to_grid(m.predict_proba(Xb), e) for m, e in zip(b["models"], b["member_edges"])], axis=0)
        Wh = Ph * self.grid_omega
        Wh /= Wh.sum(1, keepdims=True)
        xg = seq_inputs(Fw, S2 if "gru" in self.n2_into else None)
        Ps = [self.seq_mean(self.grus, (xg - b["gru_mu"]) / b["gru_sd"])]
        if "tcn" in self.comps:
            x0 = seq_inputs(Fw, S2 if "tcn" in self.n2_into else None)
            Ps.append(self.seq_mean(self.tcns, (x0 - b["tcn_mu"]) / b["tcn_sd"]))
        if "gemb" in self.comps:
            Ps.append(np.mean([self.seq_mean([g], (seq_inputs(Fw, E[rows]) - b["gemb_mu"]) / b["gemb_sd"])
                               for g, E in zip(self.gembs, self.embs)], axis=0))
        qs = np.mean([self.wq(P) for P in Ps], axis=0)
        pred = decide(self.alpha * qs + (1 - self.alpha) * Wh, None, np.ones(Wh.shape[1]), True, GRID)
        if self.gridm is not None:  # the wrapper's grid decision, elapsed from the stream start
            for i in range(len(pred)):
                pred[i] = self.gridm._grid_decide([P[i] for P in Ps], Ph[i], (i + 1) * 5.0)
        return np.clip(pred, 0, 600)

    def score(self, r, k):
        rows = r[k:]
        return float(official_rec_scores(d.iloc[rows].reset_index(drop=True), self.predict(r, k)).iloc[0])


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("--alpha-"):
            MODELS[a[8:].split("=")[0]]["alpha"] = float(a.split("=")[1])
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        path = OUT / f"shift_calib_{name}.npz"
        if path.exists():
            print(f"{name}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        run = Runner(MODELS[name])
        S = np.full((len(recs), K_MAX + 1), np.nan)
        for i, r in enumerate(recs):
            for k in range(min(K_MAX, len(r) - 1) + 1):
                S[i, k] = run.score(r, k)
        rng = [np.random.default_rng(s) for s in (0, 1, 2)]
        k_rand = np.array([[min(int(g.integers(0, max(int((o - 60) // 5), 0) + 1)), len(r) - 1) for g in rng]
                           for r, o in zip(recs, onset)])
        S_rand = np.array([[run.score(r, kk) for kk in row] for r, row in zip(recs, k_rand)])
        np.savez(path, S=S, S_rand=S_rand, k_rand=k_rand, n_windows=np.array([len(r) for r in recs]))
        print(f"{name}: k=0 {np.nanmean(S[:, 0]):.4f}, k=6 {np.nanmean(S[:, 6]):.2f}, k=12 {np.nanmean(S[:, 12]):.2f}, "
              f"random {S_rand.mean():.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
