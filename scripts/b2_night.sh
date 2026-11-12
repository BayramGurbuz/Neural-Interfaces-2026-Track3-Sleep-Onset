#!/usr/bin/env bash
# B2 overnight chain (CPU): window features once, then the B-member ablation per group in the approved order
# a (bip) -> b (events) -> c (epoch); winners among a/b combined once (c kept separate). Log: OUT/b2_night.log
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
OUT=~/neuralbench/save/cv/dual
echo "start $(date -Is)"
[ -f ~/neuralbench/save/features_b2.npz ] || $PY b2_extract.py
for g in bip events epoch; do $PY b2_ablation.py $g; done
W=$($PY -c "import json;s=json.load(open('$OUT/b2_ablation.json'));print('+'.join(g for g in ('bip','events') if s[g]['wins']))")
echo "winners among bip/events: '${W}'"
if [[ "$W" == *+* ]]; then $PY b2_ablation.py "$W"; fi
echo "done $(date -Is)"
