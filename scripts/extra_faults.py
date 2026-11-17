"""Extra fault scenarios: faults from the very first window (no clean history yet), whole-recording faults.
Checks predictions finite in [0, 600] and every piece of model state finite afterwards.
Usage: python scripts/extra_faults.py SUBMISSION_DIR   (written 15 Nov for candidate 1, kept in scripts/ from 17 Nov)"""
import copy, importlib.util, sys, types
from pathlib import Path
import joblib, numpy as np, torch
sys.path.insert(0, "/home/bayram/projects/muse-sleep-onset/scripts")
from train_hgb import SAVE
sub_dir = Path(sys.argv[1]).resolve()
stub = types.ModuleType("benchmark_utils.base_solver"); stub.CompetSolver = type("CompetSolver", (), {})
sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils"); sys.modules["benchmark_utils.base_solver"] = stub
spec = importlib.util.spec_from_file_location("submission", sub_dir / "submission.py")
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
bundle = joblib.load(sub_dir / "model.joblib")
zb = np.load(SAVE / "features_base.npz")
mt = zb["split"] == "test"
tl = zb["timeline"][mt]
X = np.asarray(np.load(SAVE / "raw_windows.npy", mmap_mode="r")[mt])
x = X[tl == tl[0]][:120].copy()
model = sub.HGBStreamModel(bundle.get("model"), 128.0, bundle.get("bin_edges"), bundle, [str(c) for c in zb["ch_names"]])

def finite_state(m):
    bad = []
    def chk(name, v):
        if v is None: return
        if isinstance(v, torch.Tensor): v = v.detach().numpy()
        if isinstance(v, (list, tuple)):
            for i, u in enumerate(v): chk(f"{name}[{i}]", u)
            return
        if isinstance(v, dict):
            for k, u in v.items(): chk(f"{name}.{k}", u)
            return
        if isinstance(v, (float, int, np.floating, np.ndarray)):
            a = np.asarray(v, dtype=np.float64)
            if a.size and not np.isfinite(a).all(): bad.append(name)
    for k in ("prev_out", "last_win", "gru_h", "gru_sum", "gemb_h", "emb_sum", "n2_prev", "n2_sum", "n2_buf"):  # n2_buf: context detector (17 Nov)
        chk(k, getattr(m, k, None))
    if getattr(m, "tcns", None) is not None:
        chk("tcn.hist", m.tcns.hist)
    chk("n2.emas", m.n2_summ.emas); chk("n2.hist", m.n2_summ.hist)
    rm = m.n2_summ.runmax
    if not np.isfinite(rm): bad.append("n2.runmax")
    for k, v in vars(m.stream).items(): chk("stream." + k, v)
    return bad

def run(xc):
    m = copy.deepcopy(model); m.reset_state()
    p = np.array([float(m.predict(torch.tensor(xc[i][None]))[0]) for i in range(len(xc))])
    return p, finite_state(m)

p0, b0 = run(x)
sc = {}
a = x.copy(); a[:5] = np.nan; sc["first 5 windows all NaN"] = a
a = x.copy(); a[0] = np.inf; sc["window 0 all +inf"] = a
a = x.copy(); a[:30, 2] = 0.0; sc["ch2 flat (0) for the first 30 windows"] = a
a = x.copy(); a[:] = np.nan; sc["whole recording NaN"] = a
a = x.copy(); a[:] = 7.0; sc["whole recording flat on all channels"] = a
a = x.copy(); a[:, 0] = np.nan; sc["ch0 NaN for the whole recording"] = a
a = x.copy(); a[10:, :, ::2] = np.nan; sc["every 2nd sample NaN from window 10"] = a
a = x.copy(); a[50] = 1e30; sc["window 50 huge finite (1e30)"] = a
a = x.copy(); a[50:60] = -np.inf; sc["windows 50-59 all -inf"] = a
ok = not b0 and np.isfinite(p0).all()
print(f"clean: finite {np.isfinite(p0).all()}, bad state {b0}")
for k, xc in sc.items():
    p, bad = run(xc)
    fin = bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 600).all())
    ok &= fin and not bad
    print(f"{k:42s} preds finite/in range {fin}; non-finite state {bad or 'none'}; max |Δ| vs clean {np.abs(p - p0).max():.1f} s; last-window |Δ| {abs(p[-1] - p0[-1]):.2f} s")
print("PASS" if ok else "FAIL")
