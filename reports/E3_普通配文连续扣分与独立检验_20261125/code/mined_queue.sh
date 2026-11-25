#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/soft_guard_20261125
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
while [[ ! -f logs/exact_test.done ]]; do sleep 8; done
for seed in a b; do
    for arm in selected_ordinary selected_joint uniform_joint; do
        "$py" -Bu code/mined_train.py "$seed" "$arm"
    done
done
touch logs/mined_train.done
for seed in a b; do
    for arm in selected_ordinary selected_joint uniform_joint; do
        "$py" -Bu code/mined_eval.py "$seed" "$arm"
    done
done
touch logs/mined_test.done
