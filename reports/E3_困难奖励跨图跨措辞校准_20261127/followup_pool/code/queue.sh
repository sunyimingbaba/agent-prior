#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/dispatch.lock
flock -n 9
until [[ -s ../collection/pool270/generation.json ]]; do sleep 6; done
bash code/run.sh &
sampler=$!
export POOL_CPU_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/private/AGSE_E1_20261110/legacy/env/bin/python -Bu code/score.py >logs/score.log 2>&1 &
scorer=$!
wait "$sampler"
wait "$scorer"
