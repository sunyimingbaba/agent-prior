#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
root=$PWD
arm=$1
py=/private/AGSE_E1_20261110/legacy/env/bin/python
while [[ ! -s repeat/$arm/eval/complete.json ]]; do sleep 10; done
for folder in "$root" "$root/repeat"; do
    cd "$folder"
    policies=("old_$arm" "$arm")
    if [[ $arm == R1 ]]; then policies=(R0 "${policies[@]}"); fi
    for policy in "${policies[@]}"; do
        "$py" -u code/training_panel.py "$policy" > "$policy.panel.log" 2>&1
    done
done
