#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
if [[ "$1" == gpu ]]; then
  while [[ ! -f logs/gpu.done ]]; do sleep 3; done
  for seed in a b; do
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/prefix_control.py "$seed"
  done
else
  while [[ ! -f logs/0.done || ! -f logs/1.done ]]; do sleep 3; done
  CUDA_VISIBLE_DEVICES='' /private/AGSE_E1_20261110/legacy/env/bin/python -u code/project_prefix.py "$1"
fi
touch "logs/prefix_$1.done"
