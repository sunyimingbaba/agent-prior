#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/strict_rfpo_20261126
while [[ ! -f ../greedy_attainable_20261126/logs/cross.done ]]; do sleep 5; done
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for seed in a b; do
    for arm in warmstrict coldordinary coldstrict; do "$py" -Bu code/train_strict.py "$seed" "$arm"; done
done
touch logs/train.done
for seed in a b; do
    for arm in warmstrict coldordinary coldstrict; do "$py" -Bu code/evaluate.py "$seed" "$arm"; done
done
touch logs/eval.done
for seed in a b; do
    for arm in warmstrict coldordinary coldstrict; do "$py" -Bu code/cross.py "$seed" "$arm"; done
done
touch logs/cross.done
