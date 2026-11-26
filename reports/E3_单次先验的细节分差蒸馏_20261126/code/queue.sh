#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/greedy_margin_20261126
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for seed in a b; do
    for arm in ordinary guard teacher; do "$py" -Bu code/fit.py "$seed" "$arm"; done
done
touch logs/train.done
for seed in a b; do
    for arm in parent ordinary guard teacher; do "$py" -Bu code/evaluate.py "$seed" "$arm"; done
done
touch logs/eval.done
