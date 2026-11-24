#!/bin/bash
set -euo pipefail
export PYTHONUTF8=1
root=$PWD
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
while [[ ! -s repeat/panel/first_R2.json ]]; do sleep 10; done
for dir in . repeat; do
 cd "$root/$dir"
 "$cpu" -u code/analyze.py --controls > controls.analysis.log 2>&1
done
touch "$root/done"
