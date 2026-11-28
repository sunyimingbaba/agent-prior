#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs records
exec 9>logs/queue.lock
flock -n 9
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
"$py" -Bu code/predict.py verify >logs/verify.log 2>&1
for tag in a_r2 a_strict b_r2 b_strict; do "$py" -Bu code/predict.py "$tag" >"logs/$tag.log" 2>&1; done
touch done
