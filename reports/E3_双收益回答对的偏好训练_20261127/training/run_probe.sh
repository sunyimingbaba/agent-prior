#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
exec 9>training/probe.lock
flock -n 9
test -f evaluation/logs/eval.done
mkdir -p logs results
for seed in a b; do
    for arm in ref r2 r3; do
        "$py" -Bu training/probe.py "$seed" "$arm" >> "logs/probe_${seed}_${arm}.log" 2>&1
    done
done
"$py" -Bu training/score_probe.py
touch logs/probe.done
