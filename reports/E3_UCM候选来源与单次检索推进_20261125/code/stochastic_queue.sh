#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
while [[ ! -f logs/cross_gpu.done ]]; do [[ $(date +%s) -lt 1795593060 ]] || exit 75; sleep 4; done
if [[ "$1" == gpu ]]; then
  for seed in a b; do /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/stochastic_deploy.py "$seed"; done
else
  CUDA_VISIBLE_DEVICES='' /private/AGSE_E1_20261110/legacy/env/bin/python -u code/stochastic_deploy.py project
fi
touch "logs/stochastic_$1.done"
