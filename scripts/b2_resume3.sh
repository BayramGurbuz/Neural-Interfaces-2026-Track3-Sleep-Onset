#!/usr/bin/env bash
# 14 Nov rule change (grid banned, Work 1 cancelled): wait for the running n2det_ens.py (HGB(c) members + n2det), then
# the Work 2 mix rows HGB(n2det) + seq(old) / seq(n2det) / seq(only GRU n2det) with the UNGRIDDED decision, alpha
# re-selected on 0.3-0.8, reference = ungridded mix3 (39.40). Acceptance: Δ <= -0.5 and >= 8/10, or Δ <= -0.3 and 10/10.
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
while pgrep -f "python n2det_ens.py" > /dev/null; do sleep 30; done
{ echo "n2det ens finished $(date -Is)"
  if [ ! -f "$OUT/b2_ens_n2det.json" ]; then echo "n2det_ens.py did not finish; rerunning"; $PY n2det_ens.py; fi
  $PY b2_mix.py "n:old,n:n,n:nG" n2nogrid nogrid; echo "n2det mix (nogrid) exit $? $(date -Is)"
  $PY - <<'PYEOF'
import json, pathlib
r = json.load(open(pathlib.Path.home() / "neuralbench/save/cv/dual/b2_mix_n2nogrid.json"))
ok = []
for k, v in r.items():
    g = v.get("nogrid") if isinstance(v, dict) else None
    if g:
        passed = (g["delta"] <= -0.5 and g["better"] >= 8) or (g["delta"] <= -0.3 and g["better"] == 10)
        print(f"{k}: macro {g['macro']:.2f} Δ {g['delta']:+.2f} (ref 39.40) better {g['better']}/10 -> {'PASS' if passed else 'no'}")
        ok.append(passed)
pathlib.Path(pathlib.Path.home() / "neuralbench/save/cv/dual/n2det_accept.txt").write_text("PASS" if any(ok) else "FAIL")
PYEOF
  echo "acceptance: $(cat $OUT/n2det_accept.txt) $(date -Is)"; } >> $OUT/n2det_ens.log 2>&1
