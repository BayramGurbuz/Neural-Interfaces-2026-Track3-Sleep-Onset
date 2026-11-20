#!/usr/bin/env bash
# Work K / Work J packages (20 Nov). Bundles from scripts/workk_bundle.py (SAVE/mix<V>_all).
#   workk_tests.sh regress      : source package folders == submitted zips; mixI_540 rebuilt with the current packager streams bit-identically
#   workk_tests.sh pkg V [EXP]  : build mix<V>_540, stream <-> batch (CPU detector reference, expected component scores EXP), then
#                                 K0 / K30: stream predictions bit-identical to mixI_540; J: equivalence with (ctx_c3 + ctx_c2 + ctxE_c2hs3p) / 3;
#                                 K / K3 / J: input-fault, 9-scenario and context / reset tests, zip SHA-256
# Log ~/neuralbench/save/workk_build.log (appended). Docker and timing separately (workk_docker.sh).
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PY=$PWD/.venv/bin/python; S=~/neuralbench/save; L=$S/workk_build.log
if [ "$1" = regress ]; then
  echo "== regress $(date -Iseconds)" >> $L
  for p in ctx_c3_540 ctx_c2_540 ctxE_c2hs3p_540 mixI_540; do
    for f in submission.py model.joblib; do unzip -p submissions/$p.zip $f | cmp -s - submissions/$p/$f && echo "$p/$f == zip" >> $L || echo "$p/$f DIFFERS from zip" >> $L; done
  done
  (cd scripts && $PY build_hgb_submission.py --model $S/mixI_all/model.joblib --name mixI_540_regK --smooth-a 0 --no-zip | tail -1) >> $L
  $PY scripts/check_stream_n2mix.py mixI_540_regK 2>&1 | tail -n 1 >> $L
  $PY -c "import numpy as np; a=np.load('submissions/mixI_540_regK_stream_preds.npy'); b=np.load('submissions/mixI_540_stream_preds.npy'); print('packager regression mixI_540: max |d|', float(np.abs(a-b).max()), 'identical', bool((a==b).all()))" >> $L
  echo "regress done $(date -Iseconds)" >> $L
  exit 0
fi
V=$2; EXP=$3; n=mix${V}_540
[ $V = J ] || L=$S/workk_build_$V.log  # K variants run in parallel: one log each
echo "== $n $(date -Iseconds)" >> $L
$PY scripts/workk_bundle.py $V 2>&1 | tail -n 1 >> $L
ZIPARG=""; case $V in K0|K30) ZIPARG="--no-zip";; esac
(cd scripts && $PY build_hgb_submission.py --model $S/mix${V}_all/model.joblib --name $n --smooth-a 0 $ZIPARG | tail -1) >> $L
$PY scripts/check_stream_n2mix.py $n $S/mix${V}_all ${EXP:+--expect=$EXP} --cpu-det 2>&1 | tail -n 11 >> $L
case $V in
  K0|K30)
    $PY -c "import numpy as np; a=np.load('submissions/${n}_stream_preds.npy'); b=np.load('submissions/mixI_540_stream_preds.npy'); print('zero-weight $V vs mixI_540 stream: max |d|', float(np.abs(a-b).max()), 'identical', bool((a==b).all()), 'PASS' if (a==b).all() else 'FAIL')" >> $L
    ;;
  *)
    if [ $V = J ]; then (cd scripts && $PY mixI_equiv_test.py $n ctx_c3_540 ctx_c2_540 ctxE_c2hs3p_540 2>&1 | tail -n 7) >> $L; fi
    $PY -c "import numpy as np; a=np.load('submissions/${n}_stream_preds.npy'); b=np.load('submissions/mixI_540_stream_preds.npy'); d=np.abs(a-b); print('info $V vs mixI_540 stream: mean |d| %.3f s, p95 %.3f s, max %.3f s' % (d.mean(), np.percentile(d, 95), d.max()))" >> $L
    (cd scripts && $PY n2det_clean_test.py ../submissions/$n 2>&1 | tail -n 4) >> $L
    (cd scripts && $PY extra_faults.py ../submissions/$n 2>&1 | tail -n 11) >> $L
    (cd scripts && $PY ctx_wrapper_test.py ../submissions/$n 2>&1 | tail -n 4) >> $L
    sha256sum submissions/$n.zip >> $L; ls -la submissions/$n.zip >> $L
    ;;
esac
echo "$n tests done $(date -Iseconds)" >> $L
