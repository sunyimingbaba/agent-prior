#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
mkdir -p logs scratch/cache/torch/kernels
py=/private/AGSE_E1_20261110/legacy/env/bin/python
for arm in a_off a_on b_off b_on; do
    if [[ ! -s policy_$arm/update.json ]]; then
        "$py" -u code/train.py "$arm" > "logs/train_$arm.log" 2>&1
    fi
done
touch logs/train_complete
