#!/bin/bash
# Work D (18 Nov): extend the 40-test shift calibration of the 8 submitted models' train+val counterparts to k = 0..120 (10 min)
cd "$(dirname "$0")"; PY=../.venv/bin/python; OUT=/home/bayram/neuralbench/save/cv/dual
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for m in final540 mix3_540 gridpi cand1; do $PY shift_calib.py $m --kmax=120 > $OUT/shift_calib120_$m.log 2>&1 & done; wait
$PY shift_calib.py cand2 --kmax=120 > $OUT/shift_calib120_cand2.log 2>&1 &
$PY shift_calib.py wb5 --alpha-wb5=0.5 --kmax=120 > $OUT/shift_calib120_wb5.log 2>&1 &
$PY shift_calib.py c2ctx --alpha-c2ctx=0.5 --kmax=120 > $OUT/shift_calib120_c2ctx.log 2>&1 &
$PY shift_calib.py c3ctx --alpha-c3ctx=0.4 --kmax=120 > $OUT/shift_calib120_c3ctx.log 2>&1 & wait
echo done
