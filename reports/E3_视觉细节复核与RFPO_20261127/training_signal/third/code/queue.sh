#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/queue.lock
flock -n 9
until [[ -f ../../extra164/done ]]; do sleep 6; done
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
native=/private/AGSE_E1_20261110/legacy/env/bin/python
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
OMP_NUM_THREADS=1 "$gpu" -Bu code/generate.py feedback >logs/feedback.log 2>&1
OMP_NUM_THREADS=1 "$gpu" -Bu code/generate.py e >logs/sampling.log 2>&1 &
sampler=$!
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 "$native" -Bu code/score.py >logs/score.log 2>&1 &
reader=$!
if ! wait "$sampler"; then kill "$reader" || true; exit 1; fi
wait "$reader"
touch done
