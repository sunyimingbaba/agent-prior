#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
mkdir -p logs training evaluation/logs evaluation/records
exec 9>training/process.lock
flock -n 9
"$py" -Bu training/prepare.py
for seed in a b; do
    "$py" -Bu training/dpo.py "$seed" ref >> "logs/reference_${seed}.log" 2>&1
done
for seed in a b; do
    for arm in r2 r3; do
        "$py" -Bu training/dpo.py "$seed" "$arm" >> "logs/train_${seed}_${arm}.log" 2>&1
    done
done
"$py" -Bu training/audit.py
touch logs/train.done
for seed in a b; do
    for arm in retained R0 r2 r3; do
        "$py" -Bu evaluation/code/predict.py "$seed" "$arm" >> "evaluation/logs/eval_${seed}_${arm}.log" 2>&1
    done
done
touch evaluation/logs/eval.done
