#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
py=/private/AGSE_E1_20261110/legacy/env/bin/python
root=$PWD
for folder in "$root" "$root/repeat"; do
    cd "$folder"
    while [[ ! -s R1/eval/complete.json || ! -s R2/eval/complete.json ]]; do sleep 10; done
    "$py" -u code/analyze.py > analyze.log 2>&1
    "$py" -u code/diagnose.py > diagnose.log 2>&1
done
