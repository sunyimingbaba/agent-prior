#!/bin/bash
set -euo pipefail
export PYTHONUTF8=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
root=$PWD
while [[ ! -s prepare.json ]]; do sleep 5; done
for dir in . repeat; do
    cd "$root/$dir"
    for a in R1 R2; do "$py" -u code/train.py "$a" > "$a.train.log" 2>&1; done
    for a in R0 R1 R2 old_R2; do
        "$py" -u code/evaluate.py "$a" > "$a.eval.log" 2>&1
    done
    "$cpu" -u code/analyze.py > analyze.log 2>&1
    "$cpu" -u code/diagnose.py > diagnose.log 2>&1
done
