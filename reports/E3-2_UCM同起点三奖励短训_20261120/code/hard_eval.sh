#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
until [[ -s logs/train_complete || -s policy_hard/update.json ]]; do sleep 10; done
for part in native generic detail score; do
    "$py" -u code/run_once.py hard "$part" >"logs/overlap_hard_${part}.log" 2>&1
done
