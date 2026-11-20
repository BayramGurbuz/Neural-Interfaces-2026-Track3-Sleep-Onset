#!/usr/bin/env bash
# Work K / J (20 Nov): Docker v3 runs in the given order on an idle machine (interleaved with candidate 1 and mixI for the cost ratio);
# every package appearing twice also gives the Docker x2 score check. Log ~/neuralbench/save/workk_docker.log (appended): name, start,
# ingestion seconds, scores. Usage: scripts/workk_docker.sh NAME1 NAME2 ...   (names of submissions/<name>.zip)
cd "$(dirname "$0")/.."
D=~/neuralbench/save/workk_docker.log
for z in "$@"; do
  echo "== $z $(date +%H:%M:%S)" >> $D
  scripts/docker_test_submission.sh submissions/$z.zip 2>&1 | grep -E "ingestion OK|FAILED|wbmae" | tr '\n' ' ' >> $D
  echo >> $D
done
echo "docker done $(date +%H:%M:%S)" >> $D
