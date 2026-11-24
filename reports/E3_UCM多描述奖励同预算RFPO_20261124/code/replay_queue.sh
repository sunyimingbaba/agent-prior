#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f records/generated.done ]; do sleep 5; done
for s in a b; do /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/replay_step.py "$s"; done
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/replay_read.py
touch records/replay.done
