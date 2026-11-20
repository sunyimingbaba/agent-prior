#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
mkdir -p logs results scratch/cache/torch/kernels
py=/private/AGSE_E1_20261110/legacy/env/bin/python
for arm in a_separate a_joint b_separate b_joint a_single b_single; do
    if [[ ! -s policy_$arm/update.json ]]; then
        "$py" -u code/train.py "$arm" > "logs/train_$arm.log" 2>&1
    fi
done
touch logs/train_complete
