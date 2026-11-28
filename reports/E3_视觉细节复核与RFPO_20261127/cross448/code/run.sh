#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs records
exec 9>logs/queue.lock
flock -n 9
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
"$py" -Bu code/predict.py lock >logs/check_inputs.log 2>&1
until [[ -f ../extra164/done && -f ../training_signal/third/e_complete.json ]]; do sleep 3; done
sleep 3
for seed in a b; do
 for arm in r2 r3; do
  "$py" -Bu code/predict.py "$seed" "$arm" >"logs/${seed}_${arm}.log" 2>&1
 done
done
touch done
