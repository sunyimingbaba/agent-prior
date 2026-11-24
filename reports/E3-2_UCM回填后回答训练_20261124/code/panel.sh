#!/bin/bash
set -euo pipefail
export PYTHONUTF8=1
root=$PWD
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
while [[ ! -s repeat/results/diagnostics.json ]]; do sleep 10; done
for dir in . repeat; do
 cd "$root/$dir"
 for a in R0 old_R2 R1 R2; do "$py" -u code/panel.py "$a" > "$a.panel.log" 2>&1; done
done
