#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f inputs/roster.json ]; do sleep 3; done
for s in a b; do
  for arm in R2 R3; do
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/fit.py "$s" "$arm"
  done
done
touch pilot/train.done
