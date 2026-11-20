#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
until [[ -f logs/train_complete && -f eval/a_on_24/complete.json && -f eval/a_off_24/complete.json ]]; do sleep 5; done
for arm in a_off a_on; do
    name=${arm}_24_force
    if [[ ! -f eval/$name/complete.json ]]; then
        "$py" -u code/evaluate.py "$arm" 24 --suppress > "logs/eval_$name.log" 2>&1
    fi
done
until [[ -f logs/eval_complete ]]; do sleep 5; done
touch logs/replicate_complete
