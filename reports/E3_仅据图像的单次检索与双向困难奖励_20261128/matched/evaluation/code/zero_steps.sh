#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs records
exec 9>logs/zero.lock
flock -n 9
until [[ -f ../training.done ]]; do sleep 3; done
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
"$py" -Bu code/zero_steps.py verify >logs/zero_verify.log 2>&1
for tag in a_R0 b_R0; do
 "$py" -Bu code/zero_steps.py "$tag" >"logs/$tag.log" 2>&1
done
touch done
