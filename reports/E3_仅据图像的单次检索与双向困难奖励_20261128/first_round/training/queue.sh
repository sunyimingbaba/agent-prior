#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/queue.lock
flock -n 9
until test -f ../matched/evaluation/done; do sleep 5; done
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for seed in a b; do
 for arm in r2 strict; do
  "$py" -Bu training/run.py "$seed" "$arm" >> "logs/train_${seed}_${arm}.log" 2>&1
 done
done
touch training.done
