"""Two EEG-free reference predictors for Track 03, fitted on train, scored on val and test.

1. constant      : one value for every window = the constant minimising train W-bMAE
                   (a weighted median of train targets; checked against a 1 s grid search).
2. elapsed-time  : prediction depends only on time elapsed since the recording started
                   (causal: = number of windows already seen * 5 s). Train windows are grouped
                   in elapsed-time bins; each bin predicts the W-bMAE-optimal value
                   (weighted median) of its train targets. A plain-mean variant is also shown.
                   Recording length / end is never used.

Usage: python scripts/simple_baselines.py [--bin-s 30]
"""
import argparse
import json

import numpy as np

from stream_utils import RANGE_NAMES, load_window_table, summarize, window_weights, weighted_median

parser = argparse.ArgumentParser()
parser.add_argument("--bin-s", type=float, default=30.0, help="elapsed-time bin width (s)")
args = parser.parse_args()

df = load_window_table()
train = df[df.split == "train"].reset_index(drop=True)
w_train = window_weights(train)

# --- 1. constant -------------------------------------------------------------
c_star = weighted_median(train.target.to_numpy(), w_train)
grid = np.arange(0, 601, 1.0)
grid_scores = [summarize(train, np.full(len(train), c))["wbmae"] for c in grid]
c_grid = float(grid[int(np.argmin(grid_scores))])
print(f"constant: weighted median = {c_star:.1f} s | grid search argmin = {c_grid:.1f} s")


# --- 2. elapsed-time rule ----------------------------------------------------
def elapsed_bin(elapsed_s: np.ndarray) -> np.ndarray:
    return np.floor((np.asarray(elapsed_s) - 1e-9) / args.bin_s).astype(int)


train_bins = elapsed_bin(train.elapsed_s.to_numpy())
n_bins = train_bins.max() + 1
table_med = np.empty(n_bins)
table_mean = np.empty(n_bins)
for b in range(n_bins):
    m = train_bins == b
    table_med[b] = weighted_median(train.target.to_numpy()[m], w_train[m])
    table_mean[b] = train.target.to_numpy()[m].mean()


def predict_elapsed(d, table):
    # later than anything seen in train -> keep the last learned value
    return table[np.minimum(elapsed_bin(d.elapsed_s.to_numpy()), n_bins - 1)]


results = {}
for split in ("train", "val", "test"):
    d = df[df.split == split].reset_index(drop=True)
    results[split] = {
        "constant": summarize(d, np.full(len(d), c_star)),
        "elapsed_wmedian": summarize(d, predict_elapsed(d, table_med)),
        "elapsed_mean": summarize(d, predict_elapsed(d, table_mean)),
    }

print(f"\nelapsed-time table (bin {args.bin_s:.0f} s, weighted median): "
      + ", ".join(f"{int(b*args.bin_s)}s:{v:.0f}" for b, v in enumerate(table_med)))
print(f"\n{'model':<18}{'split':<7}{'W-bMAE':>8}" + "".join(f"{r:>11}" for r in RANGE_NAMES))
for name in ("constant", "elapsed_wmedian", "elapsed_mean"):
    for split in ("train", "val", "test"):
        s = results[split][name]
        print(f"{name:<18}{split:<7}{s['wbmae']:>8.2f}" + "".join(f"{s[r]:>11.2f}" for r in RANGE_NAMES))

print("\nJSON:", json.dumps({"constant_value_s": c_star, "bin_s": args.bin_s, "results": results}))
