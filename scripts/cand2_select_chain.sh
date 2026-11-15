#!/usr/bin/env bash
# Candidate 2 selection (no decision taken here; results for the report). Waits for the 4-member expansion (ens_crop_cv.py c)
# and the GPU chain (cand2_gpu_chain.sh), then:
#  1. mix rows, ungridded, alpha nested LOFO 0.3-0.8, Δ vs 39.40 and vs candidate 1 (n:cropG = 38.86)    [CPU]
#  2. GRU crop + closeness final, 540 and train+val                                                         [GPU, in parallel]
#  3. candidate 2 HGB(c) final, 540 and train+val (crop + closeness)                                        [CPU]
#  4. robustness (k = 1, 3, 6, 12, 24, 60, random) of the three candidate rows with train+val models, alpha = LOFO choice
# Log OUT/cand2_select_chain.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
S=~/neuralbench/save
{ echo "waiting $(date -Is)"
  until grep -q "^expansion exit" $OUT/cand2_chain.log 2>/dev/null && grep -q "^done" $OUT/cand2_gpu_chain.log 2>/dev/null; do sleep 60; done
  echo "inputs ready $(date -Is)"; grep "exit" $OUT/cand2_chain.log $OUT/cand2_gpu_chain.log
  [ -f $OUT/ens_crop_c.json ] || { echo "expansion result missing -> stop"; exit 1; }
  ( $PY train_gru_n2c.py > $S/gru_n2c_all.log 2>&1; $PY train_gru_n2c.py --trainval > $S/gru_n2c_tv.log 2>&1; echo "GRU n2c finals done $(date -Is)" ) &
  $PY b2_mix.py "n:cropG,n2c:cropG,n2c:crop,n2c:cropGc" cand2 nogrid --ref2=n:cropG > $OUT/b2_mix_cand2.log 2>&1; echo "1 mix exit $? $(date -Is)"
  $PY train_n2det_final.py hgbmc > $S/ens_c_all_n2c_crop.log 2>&1; echo "3a HGB 540 exit $? $(date -Is)"; tail -n 2 $S/ens_c_all_n2c_crop.log
  $PY train_n2det_final.py hgbmc --trainval > $S/ens_c_tv_n2c_crop.log 2>&1; echo "3b HGB train+val exit $? $(date -Is)"; tail -n 2 $S/ens_c_tv_n2c_crop.log
  wait
  A() { $PY -c "import json; r=json.load(open('$OUT/b2_mix_cand2.json'))['HGB(n2c) + seq($1)']['alpha_nogrid']['all']; print(r if isinstance(r, float) else r)"; }
  H="--hgb-from=ens_c_tv_n2c_crop --n2-from=n2det_final_tv"
  $PY stream_start_test.py --bundle=mix3 --gru-from=mix3crop $H --alpha=$(A cropG) --tag=_cand2_cropG > $OUT/sst_cand2_cropG.log 2>&1; echo "4a exit $? $(date -Is)"
  $PY stream_start_test.py --bundle=mix3crop $H --alpha=$(A crop) --tag=_cand2_crop > $OUT/sst_cand2_crop.log 2>&1; echo "4b exit $? $(date -Is)"
  $PY stream_start_test.py --bundle=mix3 --gru-from=gru_n2c_tv $H --alpha=$(A cropGc) --tag=_cand2_cropGc > $OUT/sst_cand2_cropGc.log 2>&1; echo "4c exit $? $(date -Is)"
  echo "done $(date -Is)"; } > $OUT/cand2_select_chain.log 2>&1
