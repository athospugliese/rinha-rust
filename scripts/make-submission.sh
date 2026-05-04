#!/usr/bin/env bash
set -euo pipefail

# Stages the submission branch contents in dist/submission/.
# Run from solution-x/ root.

OUT=dist/submission
rm -rf "$OUT"
mkdir -p "$OUT"

cp docker-compose.yml "$OUT/"
cp nginx.conf "$OUT/"
cp info.json "$OUT/"

echo "Submission staged in $OUT/"
ls -la "$OUT/"
