#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
mkdir -p logs evaluation/logs evaluation/records
exec 9>training/queue.lock
flock -n 9
"$py" -Bu training/lock.py
for seed in a b; do
 "$py" -Bu training/run.py "$seed" strict >> "logs/train_${seed}_strict.log" 2>&1
done
touch logs/strict.done
for seed in a b; do
 for arm in r2 r3 strict; do
  "$py" -Bu evaluation/code/predict.py "$seed" "$arm" >> "evaluation/logs/eval_${seed}_${arm}.log" 2>&1
 done
done
touch evaluation/logs/strict.done
