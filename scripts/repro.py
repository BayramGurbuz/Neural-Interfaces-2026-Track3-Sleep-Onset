"""Deterministic GPU settings for reproducible training (call before any CUDA work).

cudnn.deterministic + no benchmark + torch.use_deterministic_algorithms(True); CUBLAS_WORKSPACE_CONFIG is
required by the latter for cuBLAS. Checked with scripts/check_determinism.py: with these settings two runs of
the CNN encoder give identical weights (without them: max |dw| 0.36); TCN is identical either way.
"""
import os


def set_deterministic():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
