#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f records/probe.done ]; do sleep 4; done
/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/message_trial.py
touch records/message.done
