#!/bin/bash
# Start only after the first-caption decision has been synchronized.
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/soft_guard_20261125
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
"$py" -c "import json; assert json.load(open('results/real_confirmation.json'))['eligible']"
while [[ ! -f logs/exact_test.done ]]; do sleep 10; done
for seed in a b; do
    for arm in R2 old new; do
        "$py" -Bu code/policy_train.py "$seed" "$arm"
    done
done
touch logs/policy_train.done
for seed in a b; do
    for arm in R2 old new; do
        "$py" -Bu code/policy_eval.py "$seed" "$arm"
    done
done
touch logs/policy_test.done
