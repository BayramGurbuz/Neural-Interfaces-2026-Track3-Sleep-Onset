#!/usr/bin/env bash
# Work C stage 3 (17 Nov): context detector through the candidate-2 pipeline. Log OUT/ctx_full_chain.log
#   GPU: GRU crop + context closeness dual CV -> final context detector (train+val, then 540)
#   CPU: HGB(c) 4-member crop expansion with context closeness summaries (dual CV)
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
OUT=~/neuralbench/save/cv/dual; PY=../.venv/bin/python
L=$OUT/ctx_full_chain.log; echo "start $(date -Iseconds)" > $L
{ $PY b2_seq_cv.py ctx6 gru --crop > $OUT/b2_seq_ctx6_gru.log 2>&1; echo "seq ctx6 gru exit $? $(date -Iseconds)" >> $L
  $PY train_n2det_final.py detctx --trainval > $OUT/detctx_tv.log 2>&1; echo "detctx tv exit $? $(date -Iseconds)" >> $L
  $PY train_n2det_final.py detctx > $OUT/detctx_all.log 2>&1; echo "detctx all exit $? $(date -Iseconds)" >> $L; } &
{ env -u OMP_NUM_THREADS $PY ens_crop_cv.py ctx6 > $OUT/ens_crop_ctx6.log 2>&1; echo "ens crop ctx6 exit $? $(date -Iseconds)" >> $L; } &
wait; echo "done $(date -Iseconds)" >> $L
