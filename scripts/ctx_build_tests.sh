#!/usr/bin/env bash
# Work C packages (18 Nov): stream/batch + component identity, input-fault tests, context-wrapper tests, Docker v3 x2 each + candidate 1
# for the time ratio. Log ~/neuralbench/save/ctx_build.log (tests), ~/neuralbench/save/ctx_docker.log (Docker)
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python; L=~/neuralbench/save/ctx_build.log; D=~/neuralbench/save/ctx_docker.log; : > $L; : > $D
for n in c2 c3; do
  E="--expect=hgb:25.0725"; [ $n = c2 ] && E="--expect=tcn:40.3806,gemb:49.0675,hgb:25.0725"
  echo "== ctx_${n}_540 $(date -Iseconds)" >> $L
  $PY scripts/check_stream_n2mix.py ctx_${n}_540 ~/neuralbench/save/ctx_${n}_all $E 2>&1 | tail -n 8 >> $L
  (cd scripts && $PY n2det_clean_test.py ../submissions/ctx_${n}_540 2>&1 | tail -n 4) >> $L
  (cd scripts && $PY extra_faults.py ../submissions/ctx_${n}_540 2>&1 | tail -n 11) >> $L
  (cd scripts && $PY ctx_wrapper_test.py ../submissions/ctx_${n}_540 2>&1 | tail -n 4) >> $L
  sha256sum submissions/ctx_${n}_540.zip >> $L
done
echo "tests done $(date -Iseconds)" >> $L
for z in ctx_c2_540 ctx_c2_540 ctx_c3_540 ctx_c3_540 mix3_n2det_cropg_540; do
  echo "== $z $(date -Iseconds)" >> $D; scripts/docker_test_submission.sh submissions/$z.zip 2>&1 | tail -n 3 >> $D
done
echo "docker done $(date -Iseconds)" >> $D
