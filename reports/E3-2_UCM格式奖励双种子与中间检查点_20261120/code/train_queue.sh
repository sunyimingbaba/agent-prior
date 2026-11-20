#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
mkdir -p logs scratch/cache/torch/kernels
py=/private/AGSE_E1_20261110/legacy/env/bin/python
for arm in a_off a_on; do
    if [[ ! -s policy_$arm/prefix3.json ]]; then
        "$py" -u code/train.py "$arm" 3 > "logs/prefix_$arm.log" 2>&1
    fi
done
"$py" -u code/prefix_check.py > logs/matched_prefix.log 2>&1
for arm in a_off a_on b_off b_on; do
    if [[ ! -s policy_$arm/update.json ]]; then
        "$py" -u code/train.py "$arm" > "logs/train_$arm.log" 2>&1
    fi
done
touch logs/train_complete
