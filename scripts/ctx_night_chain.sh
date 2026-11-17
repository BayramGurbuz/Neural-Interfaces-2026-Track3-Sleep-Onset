#!/usr/bin/env bash
# Work C (17 Nov evening -> 18 Nov): k=0 mix CV, HGB finals (540, train+val), shifted dual CV (HGB ctx on CPU, GRU ctx on GPU),
# shifted mix table. Waits for ctx_full_chain.sh (HGB crop CV) and ctx_final_gpu.sh (detector + GRU finals). Log OUT/ctx_night_chain.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
OUT=~/neuralbench/save/cv/dual; PY=../.venv/bin/python; L=$OUT/ctx_night_chain.log
DISTS=--dists=k0,k1,k2,k3,m0.75U12,m0.5U12,U6,U12,m0.5U24
echo "start $(date -Iseconds)" > $L
until grep -q "^ens crop ctx6 exit" $OUT/ctx_full_chain.log 2>/dev/null; do sleep 60; done
echo "HGB crop CV ready $(date -Iseconds)" >> $L
nice -n 5 $PY b2_mix.py "n2c:cropGc,n2c:wb5,ctx:c3ctx,ctx:c2ctx,ctx:wb5,ctx:cropGc" wc nogrid --ref2=n2c:cropGc > $OUT/b2_mix_wc.log 2>&1 &
MIXPID=$!
until grep -q "^detctx all exit" $OUT/ctx_final_gpu.log 2>/dev/null; do sleep 60; done
env -u OMP_NUM_THREADS $PY train_n2det_final.py hgbmc --det=ctx > $OUT/hgb_ctx_all.log 2>&1; echo "hgb ctx 540 exit $? $(date -Iseconds)" >> $L
until grep -q "^detctx tv exit" $OUT/ctx_final_gpu.log 2>/dev/null; do sleep 60; done
env -u OMP_NUM_THREADS $PY train_n2det_final.py hgbmc --det=ctx --trainval > $OUT/hgb_ctx_tv.log 2>&1; echo "hgb ctx tv exit $? $(date -Iseconds)" >> $L
wait $MIXPID; echo "k0 mix exit $? $(date -Iseconds)" >> $L
{ until grep -q "^done" $OUT/ctx_final_gpu.log 2>/dev/null; do sleep 60; done
  $PY shift_cv.py seq uw_n10_ctx6_crop $DISTS > $OUT/shiftcv_seq_uw_n10_ctx6_crop.log 2>&1; echo "shift seq ctx exit $? $(date -Iseconds)" >> $L; } &
env -u OMP_NUM_THREADS $PY shift_cv.py hgb --det=ctx $DISTS > $OUT/shiftcv_hgbctx.log 2>&1; echo "shift hgb ctx exit $? $(date -Iseconds)" >> $L
wait
$PY shift_cv.py mix "n2c:uw_n10_n2c_crop+tcn_n10+cnn3_emb,n2c:uw_n10_n2c_crop,ctx:uw_n10_ctx6_crop,ctx:uw_n10_ctx6_crop+tcn_n10+cnn3_emb,ctx:uw_n10_n2c_crop,ctx:uw_n10_n2c_crop+tcn_n10+cnn3_emb" $DISTS --mixout=shiftcv_mix_wc.json > $OUT/shiftcv_mix_wc.log 2>&1
echo "shift mix exit $? $(date -Iseconds)" >> $L; echo "done $(date -Iseconds)" >> $L
