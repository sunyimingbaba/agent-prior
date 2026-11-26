#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
CUDA_VISIBLE_DEVICES='' "$cpu" -Bu code/cold_sample.py project 0 >logs/cold_cpu0.log 2>&1 &
left=$!
CUDA_VISIBLE_DEVICES='' "$cpu" -Bu code/cold_sample.py project 1 >logs/cold_cpu1.log 2>&1 &
right=$!
trap 'kill "$left" "$right" 2>/dev/null || true' EXIT
for seed in a b; do "$gpu" -Bu code/cold_sample.py sample "$seed" >"logs/cold_gpu_$seed.log" 2>&1; done
wait "$left"
wait "$right"
touch logs/cold.done
