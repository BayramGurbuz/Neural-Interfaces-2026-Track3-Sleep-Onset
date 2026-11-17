#!/usr/bin/env bash
# Work C finals on the GPU (17 Nov): context detector (540, then train+val) with crop-exact close_L, then GRU crop + context
# closeness finals (540, train+val). Log OUT/ctx_final_gpu.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
OUT=~/neuralbench/save/cv/dual; PY=../.venv/bin/python; L=$OUT/ctx_final_gpu.log
echo "start $(date -Iseconds)" > $L
$PY train_n2det_final.py detctx > $OUT/detctx_all.log 2>&1; echo "detctx all exit $? $(date -Iseconds)" >> $L
$PY train_n2det_final.py detctx --trainval > $OUT/detctx_tv.log 2>&1; echo "detctx tv exit $? $(date -Iseconds)" >> $L
$PY train_gru_n2c.py --det=ctx > $OUT/gru_ctx_all.log 2>&1; echo "gru ctx all exit $? $(date -Iseconds)" >> $L
$PY train_gru_n2c.py --det=ctx --trainval > $OUT/gru_ctx_tv.log 2>&1; echo "gru ctx tv exit $? $(date -Iseconds)" >> $L
echo "done $(date -Iseconds)" >> $L
