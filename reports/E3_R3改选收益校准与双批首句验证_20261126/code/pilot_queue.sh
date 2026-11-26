#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
arms=$("$py" -c 'import json; print(" ".join(json.load(open("inputs/pilot/decision.json"))["train_arms"]))')
for seed in a b; do
  for arm in $arms; do "$py" -Bu code/train_pilot.py "$seed" "$arm" >"logs/pilot_train_${seed}_${arm}.log" 2>&1; done
done
for seed in a b; do
  for arm in $arms; do "$py" -Bu code/eval_pilot.py "$seed" "$arm" >"logs/pilot_eval_${seed}_${arm}.log" 2>&1; done
done
touch logs/pilot.done
