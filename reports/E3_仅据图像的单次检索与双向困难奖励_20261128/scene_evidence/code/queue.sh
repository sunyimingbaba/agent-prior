#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/queue.lock
flock -n 9
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
until test -f ../first_round/evaluation/done; do sleep 10; done
for tag in a_strict b_strict a_r2 b_r2; do
    "$py" -Bu code/generate.py "$tag" real >>logs/run.log 2>&1
    if [[ "$tag" == *_strict ]]; then
        "$py" -Bu code/generate.py "$tag" shuffled >>logs/run.log 2>&1
    fi
done
touch done
