#!/usr/bin/env bash
# Work G packages (19 Nov): 3-seed HGB E (ens_c_all_ctxE_s3_crop) in the two Work E packages -> ctxE_c3s3_540, ctxE_c2hs3_540; same tests
# as ctxE_build_tests.sh + Docker v3 x2 each + candidate 1. Logs ~/neuralbench/save/workg_build.log, workg_docker.log
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PY=$PWD/../.venv/bin/python; S=~/neuralbench/save; L=$S/workg_build.log; D=$S/workg_docker.log; : > $L; : > $D
until grep -q "^saved" $S/ens_c_all_ctxE_s3.log; do sleep 30; done
HV=$(grep -o "hgb_c+n2c+crop [0-9.]*" $S/ens_c_all_ctxE_s3.log | awk '{print $2}'); echo "HGB 3-seed in-sample test $HV" >> $L
$PY train_n2det_final.py mix --alpha=0.4 --ens=ens_c_all_ctxE_s3_crop --seq=mix3det_all --gru-from=gru_ctx_all --out=ctxE_c2hs3_all | tail -1 >> $L
$PY train_n2det_final.py mix --alpha=0.4 --ens=ens_c_all_ctxE_s3_crop --seq=mix3det_all --gru-from=gru_ctxE_all --out=ctxE_c3s3full_all | tail -1 >> $L
$PY - >> $L <<'PYEOF'
import joblib, json
from pathlib import Path
S = Path.home() / "neuralbench/save"
b = joblib.load(S / "ctxE_c3s3full_all/model.joblib")
b["seq_comps"] = ("gru",)
for k in ("tcn_states", "tcn_mu", "tcn_sd", "tcn_dilations", "cnn_states", "cnn_scale", "cnn_clip", "cnn_emb", "gemb_states", "gemb_mu", "gemb_sd"):
    b.pop(k, None)
out = S / "ctxE_c3s3_all"; out.mkdir(exist_ok=True); joblib.dump(b, out / "model.joblib")
(out / "results.json").write_text(json.dumps(dict(source="ctxE_c3s3full_all", mix_alpha=b["mix_alpha"], seq_comps=["gru"]), indent=2))
print("saved", out, len(b["models"]), "HGB models")
PYEOF
for n in c3s3 c2hs3; do $PY build_hgb_submission.py --model $S/ctxE_${n}_all/model.joblib --name ctxE_${n}_540 --smooth-a 0 | tail -1 >> $L; done
cd ..
for n in c3s3 c2hs3; do
  E="--expect=hgb:$HV"; [ $n = c2hs3 ] && E="--expect=tcn:40.3806,gemb:49.0675,hgb:$HV"
  echo "== ctxE_${n}_540 $(date -Iseconds)" >> $L
  $PY scripts/check_stream_n2mix.py ctxE_${n}_540 $S/ctxE_${n}_all $E --cpu-det 2>&1 | tail -n 8 >> $L
  (cd scripts && $PY n2det_clean_test.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 4) >> $L
  (cd scripts && $PY extra_faults.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 11) >> $L
  (cd scripts && $PY ctx_wrapper_test.py ../submissions/ctxE_${n}_540 2>&1 | tail -n 4) >> $L
  sha256sum submissions/ctxE_${n}_540.zip >> $L; ls -la submissions/ctxE_${n}_540.zip >> $L
done
echo "tests done $(date -Iseconds)" >> $L
for z in ctxE_c3s3_540 ctxE_c3s3_540 ctxE_c2hs3_540 ctxE_c2hs3_540 mix3_n2det_cropg_540; do
  echo "== $z $(date -Iseconds)" >> $D; scripts/docker_test_submission.sh submissions/$z.zip 2>&1 | tail -n 3 >> $D
done
echo "docker done $(date -Iseconds)" >> $D
