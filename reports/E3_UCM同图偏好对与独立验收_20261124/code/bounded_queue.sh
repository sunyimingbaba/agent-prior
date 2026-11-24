#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
while [ ! -f guard/sampled.done ]; do sleep 3; done
for s in a b; do
  /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/bound_sources.py "$s"
done
for s in a b; do
  while [ ! -f "records/${s}_bound_ref.json" ]; do sleep 3; done
  /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/bound_fit.py "$s" bounded
done
touch pilot/bound_train.done
bash code/dev_eval.sh a_bounded b_bounded
touch pilot/bound_eval.done
