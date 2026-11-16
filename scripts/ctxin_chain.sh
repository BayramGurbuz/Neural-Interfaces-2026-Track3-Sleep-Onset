#!/usr/bin/env bash
# Work C stage 2: context detector (6 windows) inner-fold outputs for 3 seeds (GPU), then the HGB "c" ablation vs the
# multi-class detector's c row (CPU, after the Work B chain). Logs OUT/n2det_ctx_inner.log, OUT/n2det_ctx_ablation.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
OUT=~/neuralbench/save/cv/dual
for s in 0 1 2; do
  ../.venv/bin/python n2det_ctx_inner.py 6 $s >> $OUT/n2det_ctx_inner.log 2>&1
  echo "seed $s exit $? $(date -Iseconds)" >> $OUT/n2det_ctx_inner.log
done
while pgrep -f wb_chain.sh > /dev/null; do sleep 60; done
../.venv/bin/python n2det_v2_ablation.py c --src=n2det_ctx6 --seeds=0,1,2 > $OUT/n2det_ctx_ablation.log 2>&1
echo "ablation exit $? $(date -Iseconds)" >> $OUT/n2det_ctx_ablation.log
