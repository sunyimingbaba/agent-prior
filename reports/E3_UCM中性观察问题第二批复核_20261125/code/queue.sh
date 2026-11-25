#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
for seed in a b; do
  /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/run.py "$seed"
done
touch records/done
