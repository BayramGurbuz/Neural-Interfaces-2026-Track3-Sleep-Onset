#!/bin/bash
# Work E finals (18 Nov night): GPU after the Work E sequence CV, CPU (HGB) after the Work E CPU chain; all deterministic.
cd "$(dirname "$0")"; PY=../.venv/bin/python; OUT=$HOME/neuralbench/save/cv/dual; L=$HOME/neuralbench/save
{ while pgrep -f "worke_gpu_chain.sh" >/dev/null; do sleep 60; done
  $PY train_gru_n2c.py --det=ctx --cropE > $L/gru_ctxE_all.log 2>&1 || echo "FAILED gru all"
  $PY train_gru_n2c.py --det=ctx --cropE --trainval > $L/gru_ctxE_tv.log 2>&1 || echo "FAILED gru tv"
  $PY train_mix3_final.py --all --deterministic --cropE --tag=cropE --epochs=gru:20,tcn:15,gemb:7 > $L/mix3cropE_all.log 2>&1 || echo "FAILED mix3 all"
  $PY train_mix3_final.py --deterministic --cropE --tag=cropE --epochs=gru:20,tcn:15,gemb:7 > $L/mix3cropE_tv.log 2>&1 || echo "FAILED mix3 tv"
  echo "gpu finals done $(date)"; } &
{ while pgrep -f "worke_cpu_chain.sh" >/dev/null; do sleep 60; done
  $PY train_n2det_final.py hgbmc --det=ctx --cropE > $L/ens_c_all_ctxE.log 2>&1 || echo "FAILED hgb all"
  $PY train_n2det_final.py hgbmc --det=ctx --cropE --trainval > $L/ens_c_tv_ctxE.log 2>&1 || echo "FAILED hgb tv"
  echo "cpu finals done $(date)"; } &
wait
