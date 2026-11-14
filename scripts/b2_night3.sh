#!/usr/bin/env bash
# B2 night 3 (14 Nov, after the sealed gridpi result): HGB(c) with the a+c (bip+epoch) group, then the mix rows
# HGB(a+c) + seq(old) and HGB(a+c) + seq(a) with the pi(D) grid decision. Log: OUT/b2_ens_bip+epoch.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
{ echo "HGB start $(date -Is)"; $PY b2_ens.py bip+epoch; echo "HGB exit $? $(date -Is)"
  echo "mix start $(date -Is)"; $PY b2_mix.py "a+c:old,a+c:a" ac; echo "mix exit $? $(date -Is)"; } > "$OUT/b2_ens_bip+epoch.log" 2>&1
