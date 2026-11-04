"""Export the elapsed-time rule (fitted on train) as the weights file of a submission.

Usage: python scripts/export_elapsed_rule.py [--bin-s 30] [--out submissions/elapsed_rule]
"""
import argparse
import json
from pathlib import Path

from stream_utils import fit_elapsed_table, load_window_table

parser = argparse.ArgumentParser()
parser.add_argument("--bin-s", type=float, default=30.0)
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "submissions" / "elapsed_rule"))
args = parser.parse_args()

df = load_window_table()
table = fit_elapsed_table(df[df.split == "train"].reset_index(drop=True), args.bin_s)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "elapsed_table.json").write_text(json.dumps(
    {"bin_s": args.bin_s, "values_s": [round(float(v), 3) for v in table],
     "fitted_on": "Interaxon2026Muse train split (412 recordings), NeuralBench _sleep_onset_stream"},
    indent=1))
print(f"wrote {out / 'elapsed_table.json'}: {len(table)} bins of {args.bin_s:.0f} s")
