#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
while [ ! -f pilot/observed_eval.done ]; do sleep 3; done
for s in a b; do
  for arm in joint continuous_native; do
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/continuous_fit.py "$s" "$arm"
  done
done
touch pilot/continuous_train.done
bash code/dev_eval.sh a_joint a_continuous_native b_joint b_continuous_native
touch pilot/continuous_eval.done
