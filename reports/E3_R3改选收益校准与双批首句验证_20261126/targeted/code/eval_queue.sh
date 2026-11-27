#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
mkdir -p logs
for seed in a b; do
  for arm in ordinary old old3; do
    until [[ -s "models/${seed}_${arm}/complete.json" ]]; do sleep 5; done
  done
done
for seed in a b; do
  for arm in retained ordinary old old3; do
    "$py" -Bu code/eval_single.py "$seed" "$arm" > "logs/eval_${seed}_${arm}.log" 2>&1
  done
done
touch logs/eval.done
