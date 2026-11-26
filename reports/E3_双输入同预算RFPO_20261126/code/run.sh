#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/dual_input_rfpo_20261126
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for seed in a b; do
    for arm in singleplain singlejoint dualplain dualjoint; do "$py" -Bu code/train_dual.py "$seed" "$arm"; done
done
touch logs/train.done
for seed in a b; do
    for arm in singleplain singlejoint dualplain dualjoint; do "$py" -Bu code/evaluate.py "$seed" "$arm"; done
done
touch logs/eval.done
for seed in a b; do
    for arm in dualplain dualjoint; do "$py" -Bu code/cross.py "$seed" "$arm"; done
done
touch logs/cross.done
