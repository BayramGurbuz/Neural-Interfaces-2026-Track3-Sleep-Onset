#!/usr/bin/env bash
# Replay Codabench ingestion + scoring locally in the worker image on a submission ZIP.
# Mirrors the platform: read-only data/submission mounts, no network.
#
# Usage: scripts/docker_test_submission.sh submissions/<name>.zip [--gpus]
#
# One-time setup (already done on this machine):
#   git clone https://github.com/neural-interfaces26/2026-competition ~/projects/2026-competition
#   C=~/neuralbench/compet; mkdir -p $C/data/neural_compet
#   cp -al ~/neuralbench/data/Interaxon2026Muse $C/data/neural_compet/      # hardlinks, no extra space
#   cp -rL ~/projects/2026-competition/codabench/phases/warmup/sleep_onset $C/input_data
#   docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $C/input_data:/app/input_data:ro \
#     -v $C/data:/app/data lionelyneuro/neural-compet:v3 sh -c 'cp -r /app/input_data/benchmark \
#     /tmp/benchmark && benchopt prepare /tmp/benchmark -d "Interaxon2026Muse[subset=full]"'
set -euo pipefail
ZIP=$(realpath "$1")
GPU_FLAG=()
[ "${2:-}" = "--gpus" ] && GPU_FLAG=(--gpus all)   # needs nvidia-container-toolkit
IMAGE=lionelyneuro/neural-compet:v3
C=~/neuralbench/compet
RUN=$C/run
rm -rf "$RUN" && mkdir -p "$RUN/ingested" "$RUN/res" "$RUN/score"
python3 -m zipfile -e "$ZIP" "$RUN/ingested"
echo "ZIP contents (must be flat, submission.py at root):"; ls -la "$RUN/ingested"

start=$(date +%s)
docker run --rm --network none "${GPU_FLAG[@]}" --user root \
  -v "$C/input_data:/app/input_data:ro" -v "$RUN/ingested:/app/ingested_program:ro" \
  -v "$C/data:/app/data:ro" -v "$RUN/res:/app/output" \
  "$IMAGE" python3 /compet/ingestion_program/ingestion.py > "$RUN/ingestion.log" 2>&1 \
  || { echo "INGESTION FAILED, see $RUN/ingestion.log"; tail -30 "$RUN/ingestion.log"; exit 1; }
echo "ingestion OK in $(( $(date +%s) - start )) s"

docker run --rm --network none --user root \
  -v "$RUN/res:/app/input/res:ro" -v "$RUN/score:/app/output" \
  "$IMAGE" python3 /compet/scoring_program/scoring.py > "$RUN/scoring.log" 2>&1 \
  || { echo "SCORING FAILED, see $RUN/scoring.log"; tail -30 "$RUN/scoring.log"; exit 1; }
echo "scores:"; cat "$RUN/score/scores.json"; echo
