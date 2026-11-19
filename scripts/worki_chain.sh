#!/bin/bash
# Work I (19 Nov): the HGB pieces needed to rebuild ctxE_c2hs3p exactly in the shifted dual CV (crop-E table, context detector):
#  R1 members B + B_h2, random_state 0 / 1 / 2 (per-seed mean of the two)  under k0, D (W24-60), D' (W0-96), U[1,6] (W12-72);
#  R2 members B_h1 + B2, random_state 0 under k0 (+ W24-60 to check 0.5 R1_s0 + 0.5 R2 == the single-seed E HGB of Work E);
# 3 draws, same shift samples as Works E/F/G; then part imix (ctx_c3, ctxE_c2h, ctxE_c2hs3p, M = 0.5 c3 + 0.5 c2hs3p, M', Work J rows).
cd "$(dirname "$0")"; PY=../.venv/bin/python; OUT=$HOME/neuralbench/save/cv/dual
B="--det=ctx --crop=E --draws=3 --tag=_i"
# 19 Nov 23:30: sequential (the parallel start saved no fold in 2 h before the machine restarted; both fits use every core)
$PY shift_cv.py hgb $B --members=B_h1,B2 --seeds=0 --dists=k0,W24-60 > $OUT/shiftcv_i_hgb_r2.log 2>&1 || echo "FAILED r2"
$PY shift_cv.py hgb $B --members=B,B_h2 --seeds=0,1,2 --dists=k0,W24-60,W0-96,W12-72 > $OUT/shiftcv_i_hgb_r1.log 2>&1 || echo "FAILED r1"
$PY shift_cv.py imix --draws=3 --dists=k0,W24-60,W0-96,W12-72 > $OUT/shiftcv_i_mix.log 2>&1 || echo "FAILED imix"
echo "worki done $(date)"
