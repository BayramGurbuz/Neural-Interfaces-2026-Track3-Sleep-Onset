#!/bin/bash
# Work G (19 Nov), pre-check: HGB member B only, crop-E table, random_state 0 / 1 / 2, shifted dual CV under D (W24-60) and D' (W0-96),
# 3 draws (same samples as Works E/F); then the HGB-only single-seed vs 3-seed-mean comparison.
cd "$(dirname "$0")"; PY=../.venv/bin/python; OUT=$HOME/neuralbench/save/cv/dual
A="--det=ctx --crop=E --members=B --seeds=0,1,2 --dists=W24-60,W0-96 --draws=3 --tag=_d"
$PY shift_cv.py hgb $A > $OUT/shiftcv_d_workg_hgb.log 2>&1 || { echo "FAILED hgb"; exit 1; }
$PY shift_cv.py seedeval $A > $OUT/shiftcv_d_workg_eval.log 2>&1 || echo "FAILED seedeval"
echo "workg done $(date)"
