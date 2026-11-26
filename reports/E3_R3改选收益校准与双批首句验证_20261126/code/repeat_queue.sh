#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for seed in a b; do
  for trial in 1 2; do
    "$py" -Bu code/repeat_first_update.py "$seed" "$trial" > "logs/replay_${seed}_${trial}.log" 2>&1
  done
done
