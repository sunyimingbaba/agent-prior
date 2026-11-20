#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
"$py" -u code/generate.py > logs/generate.log 2>&1
while [[ ! -f logs/gallery.exit ]]; do sleep 2; done
test "$(cat logs/gallery.exit)" = 0
"$py" -u code/score.py > logs/score.log 2>&1
