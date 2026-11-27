#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs models results
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
CUDA_VISIBLE_DEVICES='' "$cpu" -Bu code/parity.py >logs/parity.log 2>&1
for seed in a b; do
 for arm in ordinary old old3; do
  "$gpu" -Bu code/train.py "$seed" "$arm" >"logs/train_${seed}_${arm}.log" 2>&1
 done
done
touch logs/train.done
