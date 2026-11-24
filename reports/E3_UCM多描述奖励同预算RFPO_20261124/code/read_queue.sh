#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f pilot/train.done ]; do sleep 4; done
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/read_eval.py
touch records/read.done
