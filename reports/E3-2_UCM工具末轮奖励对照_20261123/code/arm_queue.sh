#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
py=/private/AGSE_E1_20261110/legacy/env/bin/python
root=$PWD
arm=$1
for folder in "$root" "$root/repeat"; do
    cd "$folder"
    "$py" -u code/train.py "$arm" > "$arm.train.log" 2>&1
done
for folder in "$root" "$root/repeat"; do
    cd "$folder"
    "$py" -u code/evaluate.py "$arm" > "$arm.eval.log" 2>&1
done
