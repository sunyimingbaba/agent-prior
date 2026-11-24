#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
for s in a b; do
  for arm in ref hard native; do
    if [ "$arm" = ref ] && [ -f "records/${s}_observed_ref.pt" ]; then continue; fi
    if [ -f "pilot/${s}_${arm}/complete.json" ]; then continue; fi
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/pair_fit.py "$s" "$arm"
  done
done
touch pilot/observed_train.done
