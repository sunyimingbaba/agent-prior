#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/gpu.lock
flock -n 9
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -Bu code/collect.py > logs/gpu.log 2>&1
