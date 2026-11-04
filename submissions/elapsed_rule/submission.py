"""Track 03 submission: elapsed-time rule (EEG-free reference).

Predicts the seconds to first N2 from the time elapsed since the recording started only.
Elapsed time is counted causally from the number of windows already received in the
current stream (allowed per the Scientific FAQ); the recording length or end is never used.
The lookup table (elapsed_table.json) holds, per 30 s elapsed-time bin, the W-bMAE-optimal
value (weighted median) of the train-split targets. Uses only the standard library + torch.
"""
import json

import torch
from benchmark_utils.base_solver import CompetSolver


class ElapsedTimeRule:
    def __init__(self, values_s, bin_s, sfreq, device):
        self.values_s = [float(v) for v in values_s]
        self.bin_s = float(bin_s)
        self.sfreq = float(sfreq)
        self.device = device
        self.reset_state()

    def reset_state(self):
        """Called by the evaluator on a fresh copy at the start of every recording."""
        self.elapsed_s = 0.0

    def _value(self, elapsed_s):
        b = int((elapsed_s - 1e-9) // self.bin_s)
        return self.values_s[min(max(b, 0), len(self.values_s) - 1)]

    def predict(self, X):
        # X: (B, C, T) windows of the current recording, in time order.
        window_s = X.shape[-1] / self.sfreq
        out = []
        for _ in range(X.shape[0]):
            self.elapsed_s += window_s  # elapsed time at the end of this window
            out.append(self._value(self.elapsed_s))
        return torch.tensor(out, dtype=torch.float32, device=X.device)


class Solver(CompetSolver):
    name = "ElapsedTimeRule"

    def load_model(self, meta):
        cfg = json.loads((meta["submission_dir"] / "elapsed_table.json").read_text())
        return ElapsedTimeRule(cfg["values_s"], cfg["bin_s"], meta["sfreq"], meta["device"])
