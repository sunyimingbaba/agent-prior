#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for arm in prepare a b; do "$py" -u code/sample_panel.py "$arm" > "$arm.log" 2>&1; done
touch sampling.done
