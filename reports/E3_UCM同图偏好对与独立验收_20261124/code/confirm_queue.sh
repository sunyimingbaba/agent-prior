#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
while [ ! -f guard/sampled.done ]; do sleep 4; done
/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python code/confirm_jobs.py prepare
/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/confirm_jobs.py deploy
touch confirm/greedy.done
