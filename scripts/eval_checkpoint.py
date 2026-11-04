"""Score saved EEGNet baseline checkpoints on val and test with the stream W-bMAE.

The checkpoints hold only weights (no Lightning callback state), so the best val score is
recomputed here. EEGNet is stateless across windows, so scoring all windows of a recording
in batches equals scoring them one at a time in a stream. The test score is recomputed too,
as a sanity check against NeuralBench's own test/wbmae_stream_mean.

Usage: python scripts/eval_checkpoint.py   (updates ~/neuralbench/save/eegnet_baseline/seed*.json)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralbench.experiment_config import build_experiment_configs
from neuralbench.main import Experiment
from torch.utils.data import DataLoader

from stream_utils import summarize

OUT = Path.home() / "neuralbench" / "save" / "eegnet_baseline"
device = "cuda" if torch.cuda.is_available() else "cpu"

configs = {c["seed"]: c for c in build_experiment_configs(
    "eeg", "_sleep_onset_stream", model="eegnet", quiet=True)}
loaders = None

for f in sorted(OUT.glob("seed*.json")):
    summary = json.loads(f.read_text())
    cfg = configs[summary["seed"]]
    cfg["delete_checkpoints_on_exit"] = False
    cfg["save_test_predictions"] = True
    exp = Experiment(**cfg)
    if loaders is None:  # same data for all seeds
        loaders = exp.data.prepare()
    exp.prepare_pl_module(loaders["train"], loaders["val"])
    module = exp._brain_module
    assert module.target_scaler is None, "predictions would need inverse scaling"
    state = torch.load(summary["checkpoint"], map_location="cpu", weights_only=True)
    module.load_state_dict(state["state_dict"])
    module.to(device).eval()

    for split in ("val", "test"):
        ds = loaders[split].dataset
        dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=ds.collate_fn)
        preds, timelines, targets = [], [], []
        with torch.inference_mode():
            for batch in dl:
                batch = batch.to(device)
                preds.append(module.model_forward(batch).reshape(-1).float().cpu().numpy())
                targets.append(batch.data["target"].reshape(-1).float().cpu().numpy())
                timelines += [s.timeline for s in batch.segments]
        d = pd.DataFrame({"timeline": timelines, "target": np.concatenate(targets)})
        summary[f"{split}_recomputed"] = summarize(d, np.concatenate(preds))
        print(f"seed {summary['seed']} {split}: W-bMAE {summary[f'{split}_recomputed']['wbmae']:.3f}", flush=True)

    summary["val_wbmae_best"] = summary["val_recomputed"]["wbmae"]
    f.write_text(json.dumps(summary, indent=2))
