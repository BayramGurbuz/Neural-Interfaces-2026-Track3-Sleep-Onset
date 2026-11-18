#!/usr/bin/env bash
# Work E packages (18-19 Nov): packager regression (ctx_c3_540 rebuilt with the seq_weights-capable builder -> stream predictions
# identical to the submitted package's), stream/batch + component identity (CPU detector reference), input-fault tests, context-wrapper
# tests, Docker v3 x2 each + candidate 1 for the time ratio. Logs ~/neuralbench/save/ctxE_build.log, ~/neuralbench/save/ctxE_docker.log
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PY=$PWD/.venv/bin/python; L=~/neuralbench/save/ctxE_build.log; D=~/neuralbench/save/ctxE_docker.log; : > $L; : > $D
echo "== regression $(date -Iseconds)" >> $L
cp submissions/ctx_c3_540_stream_preds.npy /tmp/claude-1000/-home-bayram-projects-muse-sleep-onset/4c1dc9c4-6982-44dc-8089-e6641f3ebac2/scratchpad/ctx_c3_ref.npy 2>/dev/null
(cd scripts && $PY build_hgb_submission.py --model ~/neuralbench/save/ctx_c3_all/model.joblib --name ctx_c3_540_regr --smooth-a 0 --no-zip | tail -1) >> $L
$PY scripts/check_stream_n2mix.py ctx_c3_540_regr 2>&1 | tail -n 1 >> $L
$PY -c "import numpy as np; a=np.load('submissions/ctx_c3_540_regr_stream_preds.npy'); b=np.load('/tmp/claude-1000/-home-bayram-projects-muse-sleep-onset/4c1dc9c4-6982-44dc-8089-e6641f3ebac2/scratchpad/ctx_c3_ref.npy'); print('regression ctx_c3: max |d|', float(np.abs(a-b).max()), 'identical', bool((a==b).all()))" >> $L
for n in c3 c2h; do
  E="--expect=hgb:29.5816"; [ $n = c2h ] && E="--expect=tcn:40.3806,gemb:49.0675,hgb:29.5816"
  echo "== ctxE_${n}_540 $(date -Iseconds)" >> $L
  $PY scripts/check_stream_n2mix.py ctxE_${n}_540 ~/neuralbench/save/ctxE_${n}_all $E --cpu-det 2>&1 | tail -n 8 >> $L
  (cd scripts && $PY n2det_clean_test.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 4) >> $L
  (cd scripts && $PY extra_faults.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 11) >> $L
  (cd scripts && $PY ctx_wrapper_test.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 4) >> $L
  sha256sum submissions/ctxE_${n}_540.zip >> $L
done
echo "tests done $(date -Iseconds)" >> $L
for z in ctxE_c3_540 ctxE_c3_540 ctxE_c2h_540 ctxE_c2h_540 mix3_n2det_cropg_540; do
  echo "== $z $(date -Iseconds)" >> $D; scripts/docker_test_submission.sh submissions/$z.zip 2>&1 | tail -n 3 >> $D
done
echo "docker done $(date -Iseconds)" >> $D
