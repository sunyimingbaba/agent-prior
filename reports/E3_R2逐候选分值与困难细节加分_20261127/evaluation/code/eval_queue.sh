#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs records
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
until test -f ../logs/train.done; do sleep 10; done
for seed in a b; do
  for arm in retained R0 r2 r3; do
    "$py" -Bu code/predict.py "$seed" "$arm" > "logs/eval_${seed}_${arm}.log" 2>&1
  done
done
touch logs/eval.done
