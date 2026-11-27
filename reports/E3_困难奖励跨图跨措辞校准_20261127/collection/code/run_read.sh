#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/reader.lock
flock -n 9
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
"$py" -Bu code/read.py 0 >logs/read0.log 2>&1 &
a=$!
"$py" -Bu code/read.py 1 >logs/read1.log 2>&1 &
b=$!
wait "$a"
wait "$b"
