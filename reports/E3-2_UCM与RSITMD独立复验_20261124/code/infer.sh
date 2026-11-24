#!/bin/bash
set -euo pipefail
export PYTHONUTF8=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for arm in prepare R0 a_R2 b_R2 a_R1 b_R1 a_first b_first; do
    "$py" -u code/infer.py "$arm" > "$arm.log" 2>&1
done
touch inference.done
