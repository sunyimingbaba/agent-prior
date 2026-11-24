#!/bin/bash
set -euo pipefail
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
while [[ ! -s results/rsitmd_b_first.pt || ! -f inference.done ]]; do sleep 10; done
"$cpu" -u code/archive.py > archive.log 2>&1
"$cpu" -u code/score.py > score.log 2>&1
"$cpu" -u code/verify.py > verify.log 2>&1
touch complete.done
