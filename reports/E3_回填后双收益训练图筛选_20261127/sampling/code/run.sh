#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
mkdir -p logs
exec 9>logs/queue.lock
flock -n 9
if [[ ! -f feedback_verified.json ]]; then
    "$py" -Bu code/generate.py feedback >logs/feedback.log 2>&1
fi
for seed in a b; do
    "$py" -Bu code/generate.py "$seed" >"logs/${seed}.log" 2>&1
done
