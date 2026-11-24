#!/bin/bash
set -euo pipefail
export PYTHONUTF8=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
root=$PWD
while [[ ! -s repeat/panel/R2.json ]]; do sleep 10; done
for dir in . repeat; do
 cd "$root/$dir"
 for task in train evaluate panel; do "$py" -u "code/$task.py" first_R2 > "first_R2.$task.log" 2>&1; done
done
