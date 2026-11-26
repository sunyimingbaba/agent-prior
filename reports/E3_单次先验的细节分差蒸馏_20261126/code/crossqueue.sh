#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/greedy_margin_20261126
while [[ ! -f logs/fitcheck.done ]]; do sleep 5; done
for s in a b; do
    for arm in parent teacher; do
        /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -Bu code/cross.py "$s" "$arm"
    done
done
touch logs/cross.done
