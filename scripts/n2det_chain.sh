#!/usr/bin/env bash
# Work 2 step e chain (starts when the CNN inner-fold run of n2det_cv.py is done; CNN detector only, cost rule).
# GPU: deterministic sequence models: GRU old (baseline), GRU / TCN / GRU-emb with n2det.  Log OUT/n2det_seq.log
# CPU: B member with n2det; if it wins (Δ <= -0.15 and >= 7/10, the B2 rule) the other three HGB(c) members.
#      Log OUT/n2det_hgb_chain.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
until [ -f $OUT/n2det_cnn_seen_f4.npy ] && ! pgrep -f "n2det_cv.py cnn" > /dev/null; do sleep 60; done
( echo "seq start $(date -Is)"
  for va in "old gru" "n2det gru" "n2det tcn" "n2det gemb"; do $PY b2_seq_cv.py $va || echo "FAILED $va"; done
  echo "seq done $(date -Is)" ) > $OUT/n2det_seq.log 2>&1 &
( echo "B start $(date -Is)"; $PY n2det_ablation.py; echo "B exit $? $(date -Is)"
  if $PY -c "import json,sys; r=json.load(open('$OUT/b2_ablation.json'))['n2det']; sys.exit(0 if r['delta'] <= -0.15 and r['better'] >= 7 else 1)"; then
    echo "B wins -> HGB(c) members $(date -Is)"; $PY n2det_ens.py; echo "ens exit $? $(date -Is)"
  else echo "B does not win -> stop HGB(c)"; fi ) > $OUT/n2det_hgb_chain.log 2>&1 &
wait
