#!/usr/bin/env bash
# Candidate 2, GPU while the CPU runs the 4-member expansion (15-16 Nov), sequentially (deterministic):
#  1. dual CV: GRU crop-augmented + multi-class closeness summaries as inputs (row "GRU crop + closeness")
#  2. final multi-class detector, 540 recordings, 3 seeds, out-of-fold closeness for the training rows
#  2b. the same detector on train+val only (robustness tests on the 40 test recordings)
#  3. final 540 TCN / CNN encoder / GRU-emb (no crop) with --deterministic (the crop-augmented trio is mix3crop_all)
# Log OUT/cand2_gpu_chain.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
S=~/neuralbench/save
{ echo "start $(date -Is)"
  $PY b2_seq_cv.py n2c gru --crop > $OUT/seq_n2c_crop.log 2>&1; echo "1 GRU crop+closeness CV exit $? $(date -Is)"; tail -n 1 $OUT/seq_n2c_crop.log
  $PY train_n2det_final.py detmc > $S/n2det_final_detmc.log 2>&1; echo "2 detector final exit $? $(date -Is)"; tail -n 1 $S/n2det_final_detmc.log
  $PY train_n2det_final.py detmc --trainval > $S/n2det_final_detmc_tv.log 2>&1; echo "2b detector train+val exit $? $(date -Is)"; tail -n 1 $S/n2det_final_detmc_tv.log
  $PY train_mix3_final.py --all --deterministic --tag=det > $S/mix3det_all_train.log 2>&1; echo "3 seq det final exit $? $(date -Is)"
  echo "done $(date -Is)"; } > $OUT/cand2_gpu_chain.log 2>&1
