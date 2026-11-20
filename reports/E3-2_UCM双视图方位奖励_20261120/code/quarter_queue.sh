#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
until [[ -f logs/train_complete ]]; do sleep 5; done
for arm in base a_single a_separate a_joint b_single b_separate b_joint; do
    step=16
    if [[ "$arm" == base ]]; then step=0; fi
    name=$(printf '%s_%02d' "$arm" "$step")
    if [[ ! -f quarter_eval/$name/complete.json ]]; then
        "$py" -u code/evaluate.py "$arm" "$step" --quarter > "logs/quarter_$name.log" 2>&1
    fi
done
touch logs/quarter_complete
