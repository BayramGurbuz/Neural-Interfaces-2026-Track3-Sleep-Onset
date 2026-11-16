#!/usr/bin/env bash
# Work B (16-17 Nov). Log OUT/wb_chain.log
#  GPU: after the TCN crop+closeness CV -> TCN closeness finals (train+val, 540) -> shifted-CV sequence parts
#  CPU: after the Work A calibration chain -> shifted-CV HGB part; then the mix rows of all distributions
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
S=~/neuralbench/save
DISTS=--dists=k0,U6,U12,U24,U60,m0.5U24,wide
{ echo "start $(date -Is)"
  ( until grep -q "^TCN crop+closeness CV exit" $OUT/shift_prep_chain.log 2>/dev/null; do sleep 30; done
    echo "GPU part starts $(date -Is)"
    $PY train_gru_n2c.py --arch=tcn --trainval > $S/tcn_n2c_tv.log 2>&1; echo "TCN n2c tv exit $? $(date -Is)"
    $PY train_gru_n2c.py --arch=tcn > $S/tcn_n2c_all.log 2>&1; echo "TCN n2c 540 exit $? $(date -Is)"
    for n in uw_n10_n2c_crop tcn_n10_n2c_crop tcn_n10 cnn3_emb cnn3_emb_old_crop; do
      $PY shift_cv.py seq $n $DISTS > $OUT/shiftcv_seq_$n.log 2>&1; echo "shift seq $n exit $? $(date -Is)"; done ) &
  until grep -q "^done" $OUT/shift_calib_chain.log 2>/dev/null; do sleep 60; done
  echo "CPU part starts $(date -Is)"
  OMP_NUM_THREADS= $PY shift_cv.py hgb $DISTS > $OUT/shiftcv_hgb.log 2>&1; echo "shift hgb exit $? $(date -Is)"
  wait
  $PY shift_cv.py mix "n2c:uw_n10_n2c_crop+tcn_n10+cnn3_emb,n2c:uw_n10_n2c_crop+tcn_n10_n2c_crop+cnn3_emb,n2c:uw_n10_n2c_crop+tcn_n10_n2c_crop+cnn3_emb_old_crop,n2c:uw_n10_n2c_crop+tcn_n10_n2c_crop,n2c:uw_n10_n2c_crop" $DISTS > $OUT/shiftcv_mix.log 2>&1
  echo "shift mix exit $? $(date -Is)"
  echo "done $(date -Is)"; } > $OUT/wb_chain.log 2>&1
