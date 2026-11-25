#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
case "$1" in
 route_gpu)
  while [[ ! -e logs/gpu.done ]]; do sleep 5; done
  for s in a b; do "$gpu" -u code/route_generation.py "$s"; done ;;
 route_cpu)
  while [[ ! -e logs/0.done || ! -e logs/1.done ]]; do sleep 5; done
  CUDA_VISIBLE_DEVICES='' "$cpu" -u code/route_projection.py ;;
 swap)
  while [[ ! -e logs/0.done || ! -e logs/1.done || ! -e logs/gpu.done ]]; do sleep 5; done
  CUDA_VISIBLE_DEVICES='' "$cpu" -u code/swap_projection.py ;;
 *) exit 2 ;;
esac
touch "logs/$1.done"
