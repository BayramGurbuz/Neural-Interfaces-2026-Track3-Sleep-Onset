#!/usr/bin/env bash
# HGB crop pre-check (B member) after the final HGB stage of candidate 1 frees the CPU. Log OUT/hgb_crop_precheck.log
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
while pgrep -f "train_n2det_final.py hgb" > /dev/null; do sleep 60; done
../.venv/bin/python hgb_crop_precheck.py > ~/neuralbench/save/cv/dual/hgb_crop_precheck.log 2>&1
