#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
case "$1" in
 gpu)
  for tag in first second; do
    for seed in a b; do "$gpu" -u code/sample_sources.py "$seed" "$tag"; done
  done ;;
 0|1) CUDA_VISIBLE_DEVICES='' "$cpu" -u code/project.py "$1" ;;
 *) exit 2 ;;
esac
touch "logs/$1.done"
