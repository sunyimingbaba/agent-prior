#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
while [ ! -f screen/records/extra_b_readouts/complete.json ]; do sleep 3; done
/private/AGSE_E1_20261110/legacy/env/bin/python code/build_rosters.py
for s in a b; do
  /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python code/contexts.py "$s"
  for arm in regular_R3 selected_R3 focused_R3 focused_R2; do
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/run_train.py "$s" "$arm"
  done
done
touch pilot/train.done
