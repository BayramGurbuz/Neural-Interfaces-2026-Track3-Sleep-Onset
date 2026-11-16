#!/usr/bin/env bash
# Work C: context detector screening after the shifted-CV sequence parts free the GPU. Log OUT/n2det_ctx.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
OUT=~/neuralbench/save/cv/dual
until grep -q "^shift seq cnn3_emb_old_crop exit" $OUT/wb_chain.log 2>/dev/null; do sleep 60; done
../.venv/bin/python n2det_ctx.py 6 12 > $OUT/n2det_ctx.log 2>&1
