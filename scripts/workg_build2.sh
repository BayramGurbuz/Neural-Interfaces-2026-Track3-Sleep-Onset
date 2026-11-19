#!/usr/bin/env bash
# Work G (19 Nov, user choice): hybrid with 3 seeds for B and B_h2 only (member weights 1/12 x 3, 1/4, 1/12 x 3, 1/4) -> ctxE_c2hs3p_540.
# Packager regression (member_weights path absent -> ctxE_c2h_540 stream predictions bit-identical), tests, Docker x2 + candidate 1.
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PY=$PWD/../.venv/bin/python; S=~/neuralbench/save; L=$S/workg_build2.log; D=$S/workg_docker2.log; : > $L; : > $D
cp ../submissions/ctxE_c2h_540_stream_preds.npy /tmp/claude-1000/-home-bayram-projects-muse-sleep-onset/4c1dc9c4-6982-44dc-8089-e6641f3ebac2/scratchpad/ctxE_c2h_ref.npy
$PY build_hgb_submission.py --model $S/ctxE_c2h_all/model.joblib --name ctxE_c2h_540_regr --smooth-a 0 --no-zip | tail -1 >> $L
(cd .. && $PY scripts/check_stream_n2mix.py ctxE_c2h_540_regr 2>&1 | tail -n 1) >> $L
$PY -c "import numpy as np; a=np.load('../submissions/ctxE_c2h_540_regr_stream_preds.npy'); b=np.load('/tmp/claude-1000/-home-bayram-projects-muse-sleep-onset/4c1dc9c4-6982-44dc-8089-e6641f3ebac2/scratchpad/ctxE_c2h_ref.npy'); print('regression ctxE_c2h: max |d|', float(np.abs(a-b).max()), 'identical', bool((a==b).all()))" >> $L
$PY build_hgb_submission.py --model $S/ctxE_c2hs3p_all/model.joblib --name ctxE_c2hs3p_540 --smooth-a 0 | tail -1 >> $L
cd ..
n=c2hs3p; echo "== ctxE_${n}_540 $(date -Iseconds)" >> $L
$PY scripts/check_stream_n2mix.py ctxE_${n}_540 $S/ctxE_${n}_all --expect=tcn:40.3806,gemb:49.0675,hgb:29.5489 --cpu-det 2>&1 | tail -n 8 >> $L
(cd scripts && $PY n2det_clean_test.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 4) >> $L
(cd scripts && $PY extra_faults.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 11) >> $L
(cd scripts && $PY ctx_wrapper_test.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 4) >> $L
sha256sum submissions/ctxE_${n}_540.zip >> $L; ls -la submissions/ctxE_${n}_540.zip >> $L
echo "tests done $(date -Iseconds)" >> $L
for z in ctxE_c2hs3p_540 ctxE_c2hs3p_540 mix3_n2det_cropg_540; do
  echo "== $z $(date -Iseconds)" >> $D; scripts/docker_test_submission.sh submissions/$z.zip 2>&1 | tail -n 3 >> $D
done
echo "docker done $(date -Iseconds)" >> $D
