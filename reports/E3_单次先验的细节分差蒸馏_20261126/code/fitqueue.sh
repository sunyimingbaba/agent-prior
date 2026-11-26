#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/greedy_margin_20261126
while [[ ! -f logs/eval.done ]]; do sleep 5; done
for seed in a b; do
    for arm in parent ordinary guard teacher; do
        /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -Bu code/fitcheck.py "$seed" "$arm"
    done
done
touch logs/fitcheck.done
