#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
for tag in "$@"; do
  s=${tag%%_*}; arm=${tag#*_}
  test -f "pilot/$tag/complete.json"
  /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/deploy_pair.py "$s" "$arm"
done
