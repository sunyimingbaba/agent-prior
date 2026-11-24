#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f records/replay.done ]; do sleep 4; done
for tag in a_R2 a_R3 b_R2 b_R3; do /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/probe_policy.py "$tag"; done
touch records/probe.done
