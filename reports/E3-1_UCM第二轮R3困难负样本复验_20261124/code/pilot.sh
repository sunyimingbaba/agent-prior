#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for seed in a b; do
    "$py" -u code/pilot_prepare.py "$seed" > "pilot/prepare_$seed.log" 2>&1
    for reward in R2 R3; do
        "$py" -u code/pilot_train.py "$seed" "$reward" > "pilot/${seed}_${reward}.log" 2>&1
    done
done
touch pilot/training.done
