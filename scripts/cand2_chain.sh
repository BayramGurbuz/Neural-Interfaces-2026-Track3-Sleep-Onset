#!/usr/bin/env bash
# Candidate 2, night chain (15 Nov): waits for the Work 3 stage 2 chain (n2det_v2_chain.sh: detector inner folds + B-member
# ablation), picks the detector by the user's rule (row cp or c wins if Δ vs B+n2det <= -0.15 and >= 7/10 folds; the better
# winning row, else the shipped detector), then runs the crop-augmented 4-member ensemble CV with it.
# Logs OUT/cand2_chain.log, OUT/ens_crop_<det>.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
{ echo "waiting for the ablation $(date -Is)"
  while pgrep -f "n2det_v2_chain.sh|n2det_v2_inner.py|n2det_v2_ablation.py" > /dev/null; do sleep 60; done
  DET=$($PY -c "
import json, pathlib
p = pathlib.Path('$OUT/n2det_v2_ablation.json')
j = json.loads(p.read_text()) if p.exists() else {}
w = {k.split('_')[-1]: v['macro'] for k, v in j.items() if v.get('wins')}
done = all(k in j for k in ('n2multi_cp', 'n2multi_c'))
print((min(w, key=w.get) if w else 'old') if done else 'INCOMPLETE')")
  echo "ablation finished $(date -Is): detector for the expansion = $DET"
  [ "$DET" = INCOMPLETE ] && { echo "ablation incomplete -> stop (no expansion started)"; exit 1; }
  $PY ens_crop_cv.py $DET > $OUT/ens_crop_$DET.log 2>&1
  echo "expansion exit $? $(date -Is)"; } > $OUT/cand2_chain.log 2>&1
