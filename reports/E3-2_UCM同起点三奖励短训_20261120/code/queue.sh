#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
for arm in old caption hard; do
    if [[ ! -s policy_$arm/update.json ]]; then
        /private/AGSE_E1_20261110/legacy/env/bin/python -u code/train.py "$arm" > "logs/train_$arm.log" 2>&1
    fi
done
touch logs/train_complete
