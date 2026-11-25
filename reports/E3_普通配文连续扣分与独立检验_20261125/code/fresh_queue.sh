#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
if [[ $1 == gpu ]]; then
  "$gpu" -u code/fresh_remote.py feedback
  for s in a b; do "$gpu" -u code/fresh_remote.py sample "$s"; done
else
  CUDA_VISIBLE_DEVICES='' "$cpu" -u code/fresh_remote.py project "$1"
fi
touch "logs/$1.done"
