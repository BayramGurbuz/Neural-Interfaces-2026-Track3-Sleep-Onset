#!/bin/bash
# Work E (18 Nov), GPU: shifted dual CV of the three sequence models trained with the Work E crop mix (10 % none, 50 % U[2,5] min,
# 40 % U[0,10] min; per epoch and recording; deterministic), held-out under W24-60 / W0-96 / W12-72, 3 draws (same samples as Work F).
cd "$(dirname "$0")"; PY=../.venv/bin/python; OUT=$HOME/neuralbench/save/cv/dual
A="--dists=W24-60,W0-96,W12-72 --draws=3 --tag=_d"
for s in uw_n10_ctx6_cropE tcn_n10_cropE cnn3_emb_cropE; do $PY shift_cv.py seq $s $A > $OUT/shiftcv_d_seq_$s.log 2>&1 || echo "FAILED $s"; done
echo "worke gpu done $(date)"
