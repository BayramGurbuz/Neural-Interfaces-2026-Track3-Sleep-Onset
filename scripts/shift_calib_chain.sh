#!/usr/bin/env bash
# Work A: shift calibration of the 5 submitted models (train+val counterparts). Log OUT/shift_calib_chain.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
{ echo "start $(date -Is)"
  for m in final540 mix3_540 gridpi; do $PY shift_calib.py $m > $OUT/shift_calib_$m.log 2>&1 & done
  until grep -q "^cand2 tv exit" $OUT/shift_prep_chain.log 2>/dev/null; do sleep 30; done
  grep "exit" $OUT/shift_prep_chain.log
  for m in cand1 cand2; do $PY shift_calib.py $m > $OUT/shift_calib_$m.log 2>&1 & done
  wait
  for m in final540 mix3_540 gridpi cand1 cand2; do tail -n 1 $OUT/shift_calib_$m.log; done
  echo "done $(date -Is)"; } > $OUT/shift_calib_chain.log 2>&1
