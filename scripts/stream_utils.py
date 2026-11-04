"""Shared helpers: window table from NeuralBench and the Track 03 W-bMAE metric.

W-bMAE (per recording): MAE inside true-target ranges [0,40), [40,90), [90,300), [300,600]
(600 included in the last), combined over non-empty ranges as sum(w*MAE)/sum(w) with
w = 10, 5, 3, 1. The reported score is the mean over recordings (= NeuralBench's
``wbmae_stream_mean``).
"""
from pathlib import Path

import numpy as np
import pandas as pd

EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0])
WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
RANGE_NAMES = ["[0,40)", "[40,90)", "[90,300)", "[300,600]"]
CAP_S = 600.0
WINDOWS_CACHE = Path.home() / "neuralbench" / "save" / "sleep_onset_stream_windows.parquet"


def load_window_table(use_cache: bool = True) -> pd.DataFrame:
    """One row per 5 s window, exactly as NeuralBench builds them for ``_sleep_onset_stream``.

    Columns: split, timeline, subject, stop, elapsed_s, target.
    ``elapsed_s`` is the time since the recording started (= n_windows_seen * 5 s): causal.
    The recording length / end is NOT stored.
    """
    if use_cache and WINDOWS_CACHE.exists():
        return pd.read_parquet(WINDOWS_CACHE)
    from neuralbench.experiment_config import build_experiment_configs
    from neuralbench.main import Experiment

    cfg = build_experiment_configs("eeg", "_sleep_onset_stream", model="eegnet", quiet=True)[0]
    loaders = Experiment(**cfg).data.prepare()
    rows = []
    for split in ("train", "val", "test"):
        for seg in loaders[split].dataset.segments:
            trig = seg.trigger
            n2 = float(trig._get_field_or_extra("n2_onset"))
            rows.append(
                dict(
                    split=split,
                    timeline=seg.timeline,
                    stop=seg.stop,
                    elapsed_s=seg.stop - trig.start,
                    target=float(np.clip(n2 - seg.stop, 0.0, CAP_S)),
                )
            )
    df = pd.DataFrame(rows)
    df["subject"] = df.timeline.str.extract(r"subject=([^,]+)")[0]
    df = df.sort_values(["split", "timeline", "stop"]).reset_index(drop=True)
    WINDOWS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WINDOWS_CACHE)
    return df


def range_index(target: np.ndarray) -> np.ndarray:
    """Range index 0..3 for each true target ([lo,hi), last range includes 600)."""
    return np.clip(np.searchsorted(EDGES, target, side="right") - 1, 0, len(WEIGHTS) - 1)


def per_recording_scores(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Per-recording W-bMAE plus per-range MAE (NaN where a recording has no window in it)."""
    d = df[["timeline", "target"]].copy()
    d["err"] = np.abs(np.asarray(pred, dtype=float) - d.target.to_numpy())
    d["rng"] = range_index(d.target.to_numpy())
    mae = d.groupby(["timeline", "rng"]).err.mean().unstack()  # rec x range
    mae = mae.reindex(columns=range(len(WEIGHTS)))
    w = np.where(mae.notna(), WEIGHTS, 0.0)
    out = pd.DataFrame(mae.to_numpy(), index=mae.index, columns=RANGE_NAMES)
    out["wbmae"] = np.nansum(mae.to_numpy() * w, axis=1) / w.sum(axis=1)
    return out


def summarize(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """W-bMAE mean over recordings + per-range MAE averaged over recordings having that range."""
    rec = per_recording_scores(df, pred)
    return {"wbmae": float(rec.wbmae.mean()), **{r: float(rec[r].mean()) for r in RANGE_NAMES}}


def window_weights(df: pd.DataFrame) -> np.ndarray:
    """Weights w_i such that sum_i w_i*|pred_i - target_i| equals the W-bMAE recording mean.

    W-bMAE is linear in the absolute errors once the true targets are fixed, so the best
    constant (or best value inside any group of windows) is a weighted median.
    """
    rng = range_index(df.target.to_numpy())
    key = pd.DataFrame({"tl": df.timeline.to_numpy(), "rng": rng})
    n_in_range = key.groupby(["tl", "rng"]).rng.transform("size").to_numpy()
    present = key.drop_duplicates()
    present["w"] = WEIGHTS[present.rng.to_numpy()]
    w_sum = present.groupby("tl").w.sum()
    n_rec = df.timeline.nunique()
    return WEIGHTS[rng] / (n_in_range * w_sum.loc[key.tl].to_numpy() * n_rec)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def fit_elapsed_table(train: pd.DataFrame, bin_s: float = 30.0) -> np.ndarray:
    """Elapsed-time rule: per elapsed-time bin, the W-bMAE-optimal (weighted median) target.

    Bin b covers elapsed time (b*bin_s, (b+1)*bin_s]; elapsed = time since the recording
    started at the end of the current window. Later bins than seen in train reuse the last.
    """
    w = window_weights(train)
    bins = np.floor((train.elapsed_s.to_numpy() - 1e-9) / bin_s).astype(int)
    tgt = train.target.to_numpy()
    return np.array([weighted_median(tgt[bins == b], w[bins == b]) for b in range(bins.max() + 1)])
