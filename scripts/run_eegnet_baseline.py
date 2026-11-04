"""Full NeuralBench EEGNet baseline on _sleep_onset_stream, default seeds (33, 34, 35).

Runs from Python so the best checkpoint is kept (delete_checkpoints_on_exit=False) and
per-window test predictions are saved (for the per-range errors). One JSON per seed is
written to ~/neuralbench/save/eegnet_baseline/.

Usage: python scripts/run_eegnet_baseline.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import torch
from neuralbench.experiment_config import build_experiment_configs
from neuralbench.main import Experiment

from stream_utils import summarize

OUT = Path.home() / "neuralbench" / "save" / "eegnet_baseline"
OUT.mkdir(parents=True, exist_ok=True)


def best_checkpoint(exp: Experiment) -> Path | None:
    for folder in (exp.infra.uid_folder(), Path(exp.infra.folder)):
        if folder is None:
            continue
        hits = sorted(Path(folder).rglob("best*.ckpt"), key=lambda p: p.stat().st_mtime)
        hits = [h for h in hits if f"seed={exp.seed}" in str(h)] or hits
        if hits:
            return hits[-1]
    return None


configs = build_experiment_configs("eeg", "_sleep_onset_stream", model="eegnet", quiet=True)
print(f"{len(configs)} configs, seeds: {[c['seed'] for c in configs]}", flush=True)

for cfg in configs:
    cfg["delete_checkpoints_on_exit"] = False
    cfg["save_test_predictions"] = True
    exp = Experiment(**cfg)
    print(f"=== seed {exp.seed} ===", flush=True)
    t0 = time.perf_counter()
    res = exp.run()
    wall_s = time.perf_counter() - t0

    preds = exp.test_predictions()
    d = pd.DataFrame({"timeline": preds["metadata"]["timeline"].to_numpy(),
                      "target": preds["y_true"].reshape(-1)})
    ours = summarize(d, preds["y_pred"].reshape(-1))

    ckpt = best_checkpoint(exp)
    val_best, best_epoch = None, None
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        best_epoch = state.get("epoch")
        for key, cb in state.get("callbacks", {}).items():
            if key.startswith("ModelCheckpoint") and cb.get("best_model_score") is not None:
                val_best = float(cb["best_model_score"])

    summary = {
        "seed": exp.seed,
        "val_wbmae_best": val_best,
        "best_epoch": best_epoch,
        "test_wbmae_neuralbench": float(res["test/wbmae_stream_mean"]),
        "test_ours": ours,
        "training_time_s": res.get("training_time_s"),
        "wall_time_s": wall_s,
        "peak_gpu_memory_mb": res.get("peak_gpu_memory_mb"),
        "checkpoint": str(ckpt) if ckpt else None,
        "all_test_metrics": {k: float(v) for k, v in res.items()
                             if k.startswith("test/") and v is not None},
    }
    (OUT / f"seed{exp.seed}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
