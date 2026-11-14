#!/usr/bin/env bash
# 14 Nov: stream-start augmentation of the sequence models (old 62-feature inputs), deterministic, dual CV. Log OUT/seq_crop.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
{ echo "seq crop start $(date -Is)"
  for a in gru tcn gemb; do $PY b2_seq_cv.py old $a --crop || echo "FAILED $a"; done
  echo "seq crop done $(date -Is)"; } > $OUT/seq_crop.log 2>&1
