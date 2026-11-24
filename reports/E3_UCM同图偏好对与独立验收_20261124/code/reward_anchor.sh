#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
while [ ! -f reward_confirm/sampled.done ]; do sleep 4; done
/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/reward_anchor_jobs.py deploy
while [ ! -f reward_confirm/scored.done ]; do sleep 4; done
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/reward_anchor_jobs.py read
touch reward_confirm/anchor.done
