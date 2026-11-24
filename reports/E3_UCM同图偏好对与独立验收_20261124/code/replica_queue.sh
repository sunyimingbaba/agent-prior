#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f reward_confirm/anchor.done ]; do sleep 4; done
/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/reward_anchor_jobs.py deploy bounded
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/reward_anchor_jobs.py read bounded
touch reward_confirm/replica.done
