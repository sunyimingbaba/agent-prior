#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/private/AGSE_E1_20261110/legacy/env/bin/python
"$py" -Bu code/read_native.py 0 > read0.log 2>&1 & p=$!
"$py" -Bu code/read_native.py 1 > read1.log 2>&1 & q=$!
wait "$p"
wait "$q"
touch results/native_done
