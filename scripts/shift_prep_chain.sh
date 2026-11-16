#!/usr/bin/env bash
# 16 Nov: Work A inputs (candidate 1 train+val counterpart) and Work B GPU CV. Log OUT/shift_prep_chain.log
#  GPU: binary detector train+val -> TCN crop + closeness dual CV
#  CPU: after the detector, HGB(c)+n2det train+val, then the candidate-1 train+val bundle
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
S=~/neuralbench/save
{ echo "start $(date -Is)"
  $PY train_n2det_final.py det --trainval > $S/n2det_final_det_tv.log 2>&1; echo "det tv exit $? $(date -Is)"; tail -n 1 $S/n2det_final_det_tv.log
  ( $PY b2_seq_cv.py n2c tcn --crop > $OUT/seq_tcn_n2c_crop.log 2>&1; echo "TCN crop+closeness CV exit $? $(date -Is)"; tail -n 1 $OUT/seq_tcn_n2c_crop.log ) &
  $PY train_n2det_final.py hgb --trainval > $S/ens_c_tv_n2det.log 2>&1; echo "HGB n2det tv exit $? $(date -Is)"; tail -n 2 $S/ens_c_tv_n2det.log
  $PY train_n2det_final.py mix --alpha=0.6 --ens=ens_c_tv_n2det --seq=mix3 --gru-from=mix3crop --out=cand1_tv > $S/cand1_tv.log 2>&1; echo "cand1 tv exit $? $(date -Is)"
  $PY train_n2det_final.py mix --alpha=0.6 --ens=ens_c_tv_n2c_crop --seq=mix3_det --gru-from=gru_n2c_tv --out=cand2_tv > $S/cand2_tv.log 2>&1; echo "cand2 tv exit $? $(date -Is)"
  wait
  echo "done $(date -Is)"; } > $OUT/shift_prep_chain.log 2>&1
