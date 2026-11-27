#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
exec 9>queue.lock
flock -n 9
until test -f ../evaluation/logs/strict.done; do sleep 10; done
for arm in rfpo_r2 rfpo_strict dpo_r2 dpo_r3; do
 for seed in a b; do
  "$py" -Bu code/predict.py "$seed" "$arm"
 done
done
touch done
