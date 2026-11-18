#!/bin/bash
# Work E (18 Nov), CPU: waits for the Work F HGB run (one CPU-heavy HGB at a time), then the Work E HGB (2 crop copies + ~10 % original
# rows) under the same shifted samples, then (after the GPU chain) the mix rows. First row = ctx_c3 recipe (Work E reference).
cd "$(dirname "$0")"; PY=../.venv/bin/python; OUT=$HOME/neuralbench/save/cv/dual
A="--dists=W24-60,W0-96,W12-72 --draws=3 --tag=_d"
while pgrep -f "shift_cv.py hgb --det=ctx --dists" >/dev/null; do sleep 60; done
$PY shift_cv.py hgb --det=ctx --crop=E $A > $OUT/shiftcv_d_hgbctxE.log 2>&1 || { echo "FAILED hgb E"; exit 1; }
while pgrep -f "worke_gpu_chain.sh" >/dev/null; do sleep 60; done
ROWS="ctx:uw_n10_ctx6_crop,ctx:uw_n10_ctx6_crop+tcn_n10+cnn3_emb,ctxE:uw_n10_ctx6_cropE,ctxE:uw_n10_ctx6_cropE+tcn_n10_cropE+cnn3_emb_cropE"
ROWS="$ROWS,ctxE:uw_n10_ctx6_crop,ctx:uw_n10_ctx6_cropE,ctxE:uw_n10_ctx6_crop+tcn_n10+cnn3_emb"
$PY shift_cv.py mix $ROWS $A --mixout=shiftcv_d_mix_e.json > $OUT/shiftcv_d_mix_e.log 2>&1 || echo "FAILED mix"
echo "worke cpu done $(date)"
