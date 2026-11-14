#!/usr/bin/env bash
# 14 Nov, after the user's instructions: wait for the running n2det_ens.py (HGB(c) members + n2det), then the Work 2 mix
# rows HGB(n2det) + seq(old) / seq(n2det) / seq(only GRU n2det) with alpha re-selected on 0.3-0.8 (b2_mix.py).
# Acceptance: macro Δ <= -0.5 and >= 8/10 folds, or Δ <= -0.3 and 10/10. Passed -> stop (final pipeline by hand,
# Work 1 cancelled). Failed -> Work 1 (HGB(c) a+c members + mix rows). Logs: OUT/n2det_ens.log, OUT/b2_ens_bip+epoch.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
while pgrep -f "python n2det_ens.py" > /dev/null; do sleep 30; done
{ echo "n2det ens finished $(date -Is)"
  if [ ! -f "$OUT/b2_ens_n2det.json" ]; then echo "n2det_ens.py did not finish; rerunning"; $PY n2det_ens.py; fi
  $PY b2_mix.py "n:old,n:n,n:nG" n2full; echo "n2det mix exit $? $(date -Is)"
  $PY - <<'PYEOF'
import json, pathlib
r = json.load(open(pathlib.Path.home() / "neuralbench/save/cv/dual/b2_mix_n2full.json"))
ok = []
for k, v in r.items():
    g = v.get("grid") if isinstance(v, dict) else None
    if g:
        passed = (g["delta"] <= -0.5 and g["better"] >= 8) or (g["delta"] <= -0.3 and g["better"] == 10)
        print(f"{k}: macro {g['macro']:.2f} Δ {g['delta']:+.2f} better {g['better']}/10 -> {'PASS' if passed else 'no'}")
        ok.append(passed)
pathlib.Path(pathlib.Path.home() / "neuralbench/save/cv/dual/n2det_accept.txt").write_text("PASS" if any(ok) else "FAIL")
PYEOF
} >> $OUT/n2det_ens.log 2>&1
if [ "$(cat $OUT/n2det_accept.txt)" = "PASS" ]; then echo "ACCEPTED -> Work 1 cancelled $(date -Is)" >> $OUT/n2det_ens.log; exit 0; fi
{ echo "HGB start $(date -Is)"; $PY b2_ens.py bip+epoch; echo "HGB exit $? $(date -Is)"
  $PY b2_mix.py "a+c:old,a+c:a" ac; echo "mix exit $? $(date -Is)"; } >> "$OUT/b2_ens_bip+epoch.log" 2>&1
