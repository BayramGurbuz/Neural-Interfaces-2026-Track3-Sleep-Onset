#!/usr/bin/env bash
# B2 night 2. CPU: HGB(c) with the bipolar group after the running bip+epoch B-member ablation finishes.
# GPU (parallel): sequence models with bip and bip+epoch inputs, deterministic. Logs: OUT/b2_ens_bip.log, OUT/b2_seq.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
( while pgrep -f "b2_ablation.py bip\+epoch" > /dev/null; do sleep 60; done
  echo "HGB start $(date -Is)"; $PY b2_ens.py bip; echo "HGB exit $? $(date -Is)" ) > $OUT/b2_ens_bip.log 2>&1 &
( echo "seq start $(date -Is)"
  for v in bip bip+epoch; do for a in gru tcn gemb; do $PY b2_seq_cv.py $v $a || echo "FAILED $v $a"; done; done
  echo "seq done $(date -Is)" ) > $OUT/b2_seq.log 2>&1 &
wait
