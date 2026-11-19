#!/usr/bin/env bash
# Work I package mixI_540 (19 Nov): packager regressions, folder == submitted zip, stream/batch + component identity (CPU detector),
# equivalence with the two submitted packages, input-fault and context tests, Docker v3 x2 (scores only; timing measured separately on an
# idle machine). Logs ~/neuralbench/save/mixI_build.log, mixI_docker.log
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PY=$PWD/.venv/bin/python; S=~/neuralbench/save; L=$S/mixI_build.log; D=$S/mixI_docker.log; : > $L; : > $D
for p in ctx_c3_540 ctxE_c2hs3p_540; do
  for f in submission.py model.joblib; do unzip -p submissions/$p.zip $f | cmp -s - submissions/$p/$f && echo "$p/$f == zip" >> $L || echo "$p/$f DIFFERS from zip" >> $L; done
done
for p in ctx_c3 ctxE_c2hs3p; do
  src=$([ $p = ctx_c3 ] && echo ctx_c3_all || echo ctxE_c2hs3p_all)
  (cd scripts && $PY build_hgb_submission.py --model $S/$src/model.joblib --name ${p}_540_regI --smooth-a 0 --no-zip | tail -1) >> $L
  $PY scripts/check_stream_n2mix.py ${p}_540_regI 2>&1 | tail -n 1 >> $L
  $PY -c "import numpy as np; a=np.load('submissions/${p}_540_regI_stream_preds.npy'); b=np.load('submissions/${p}_540_stream_preds.npy'); print('regression ${p}: max |d|', float(np.abs(a-b).max()), 'identical', bool((a==b).all()))" >> $L
done
n=mixI_540; echo "== $n $(date -Iseconds)" >> $L
$PY scripts/check_stream_n2mix.py $n $S/mixI_all --expect=gru:32.1071,tcn:40.3806,gemb:49.0675,hgb1:29.5489,hgb2:25.0725 --cpu-det 2>&1 | tail -n 11 >> $L
(cd scripts && $PY mixI_equiv_test.py mixI_540 ctx_c3_540 ctxE_c2hs3p_540 0.5 2>&1 | tail -n 6) >> $L
(cd scripts && $PY n2det_clean_test.py ../submissions/$n 2>&1 | tail -n 4) >> $L
(cd scripts && $PY extra_faults.py ../submissions/$n 2>&1 | tail -n 11) >> $L
(cd scripts && $PY ctx_wrapper_test.py ../submissions/$n 2>&1 | tail -n 4) >> $L
sha256sum submissions/$n.zip >> $L; ls -la submissions/$n.zip >> $L
echo "tests done $(date -Iseconds)" >> $L
for z in mixI_540 mixI_540; do echo "== $z $(date -Iseconds)" >> $D; scripts/docker_test_submission.sh submissions/$z.zip 2>&1 | tail -n 1 >> $D; done
echo "docker done $(date -Iseconds)" >> $D
