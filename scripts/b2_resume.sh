#!/usr/bin/env bash
# Resume after the 14 Nov shutdown (HGB member runs lost; per-protocol checkpoints added). Sequential on the CPU:
# Work 2 first (HGB(c) + n2det, 3 members, then mix rows), then Work 1 (HGB(c) a+c, 3 members, then mix rows).
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
{ echo "n2det ens start $(date -Is)"; $PY n2det_ens.py; echo "n2det ens exit $? $(date -Is)"
  $PY b2_mix.py "n:old,n:n" n2full; echo "n2det mix exit $? $(date -Is)"; } > $OUT/n2det_ens.log 2>&1
{ echo "HGB start $(date -Is)"; $PY b2_ens.py bip+epoch; echo "HGB exit $? $(date -Is)"
  $PY b2_mix.py "a+c:old,a+c:a" ac; echo "mix exit $? $(date -Is)"; } > "$OUT/b2_ens_bip+epoch.log" 2>&1
