#!/bin/bash
# Work F (18 Nov): shifted dual CV under the Work D distributions (D = U[2,5] min = W24-60; D' = U[0,8] min = W0-96; info U[1,6] = W12-72),
# 3 draws, drop convention; ctx HGB (CPU) || the 3 sequence models (GPU), then the w * ctx_c2 + (1 - w) * ctx_c3 mixes.
# Outputs OUT/shiftcv_d_*.npz, OUT/shiftcv_d_fmix.json (also the Work E reference: the ctx_c3 recipe under D).
cd "$(dirname "$0")"; PY=../.venv/bin/python; OUT=$HOME/neuralbench/save/cv/dual
A="--dists=W24-60,W0-96,W12-72 --draws=3 --tag=_d"
$PY shift_cv.py hgb --det=ctx $A > $OUT/shiftcv_d_hgbctx.log 2>&1 &
{ for s in uw_n10_ctx6_crop tcn_n10 cnn3_emb; do $PY shift_cv.py seq $s $A > $OUT/shiftcv_d_seq_$s.log 2>&1; done; } &
wait
$PY shift_cv.py fmix $A > $OUT/shiftcv_d_fmix.log 2>&1
echo "workf done $(date)"
