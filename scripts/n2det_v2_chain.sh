#!/usr/bin/env bash
# Work 3 stage 2 (15 Nov): GPU inner-fold outputs of the multi-class detector, seeds 0 and 1 together, then 2 (4 GB VRAM);
# then the CPU B-member ablation (rows cp, c). Logs OUT/n2det_multi_s<seed>.log, OUT/n2det_v2_ablation.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
$PY n2det_v2_inner.py 0 > $OUT/n2det_multi_s0.log 2>&1 &
$PY n2det_v2_inner.py 1 > $OUT/n2det_multi_s1.log 2>&1 &
wait -n
$PY n2det_v2_inner.py 2 > $OUT/n2det_multi_s2.log 2>&1
wait
for s in 0 1 2; do grep -q "^seed $s done" $OUT/n2det_multi_s$s.log || { echo "seed $s FAILED" > $OUT/n2det_v2_ablation.log; exit 1; }; done
$PY n2det_v2_ablation.py cp c > $OUT/n2det_v2_ablation.log 2>&1
