"""Reproducibility of the mix3 GPU components: train the TCN (seed 0, 10 epochs) and the CNN encoder (seed 0,
12 epochs) on all 540 recordings exactly like train_mix3_final.py --all, twice, and compare the weights
with each other and with the shipped bundle (~/neuralbench/save/mix3_all/model.joblib).

  --deterministic : cudnn.deterministic = True, cudnn.benchmark = False, torch.use_deterministic_algorithms(True)
                    (needs CUBLAS_WORKSPACE_CONFIG=:4096:8, set here before CUDA initialises)
  default         : the settings mix3_540 was trained with (PyTorch defaults)

Usage: python scripts/check_determinism.py [--deterministic]
"""
import os
import sys

DET = "--deterministic" in sys.argv
if DET:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

if DET:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

from cnn_encoder import RAW  # noqa: E402
from cnn_encoder import train_fold as train_encoder  # noqa: E402
from dist_hgb import VARIANTS, to_class  # noqa: E402
from gru_cv import CONFIGS, DEV, load_inputs, run_fold  # noqa: E402
from train_hgb import SAVE  # noqa: E402

print(f"deterministic={DET}, cudnn.benchmark={torch.backends.cudnn.benchmark}, "
      f"cudnn.deterministic={torch.backends.cudnn.deterministic}, device {DEV}", flush=True)
d, recs, F, gain = load_inputs(include_test=True)
tr = np.ones(len(d), bool)
X_raw = torch.as_tensor(np.asarray(np.load(RAW, mmap_mode="r")), device=DEV)
y_cls = torch.as_tensor(to_class(d.target.to_numpy(), VARIANTS["B"]), device=DEV).long()
shipped = joblib.load(SAVE / "mix3_all" / "model.joblib")


def states(m):
    return {k: v.detach().cpu().numpy().copy() for k, v in m.state_dict().items()}


def maxdiff(a, b):
    return max(float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in a)


runs = {"tcn": [], "cnn": []}
for rep in range(2):
    m, _, _ = run_fold(d, recs, F, gain, tr, None, CONFIGS["tcn_n10"], 0, stop_epoch=10)
    runs["tcn"].append(states(m))
    runs["cnn"].append(states(train_encoder(X_raw, y_cls, tr, None, 0, stop_epoch=12)))
    print(f"run {rep + 1} done", flush=True)
for name, ref in (("tcn", shipped["tcn_states"][0]), ("cnn", shipped["cnn_states"][0])):
    a, b = runs[name]
    print(f"{name} seed 0: run1 vs run2 max |Δw| = {maxdiff(a, b):.3e}; run1 vs shipped mix3_all = {maxdiff(a, ref):.3e}; "
          f"run2 vs shipped = {maxdiff(b, ref):.3e}", flush=True)
