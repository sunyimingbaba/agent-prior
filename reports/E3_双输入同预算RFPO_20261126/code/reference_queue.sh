#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/dual_input_rfpo_20261126
until test -f logs/cross.done; do sleep 10; done
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for seed in a b; do "$py" -Bu code/reference_run.py "$seed" retained; done
touch logs/reference.done
"$py" -B code/reference_archive.py
touch logs/reference_archive.done
