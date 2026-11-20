#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=4 PYTHONUTF8=1 LC_ALL=C.UTF-8
py=/private/AGSE_E1_20261110/legacy/env/bin/python
until [[ -f samples/original/2066/sample7.json ]]; do sleep 5; done
"$py" -u code/score_split.py train
"$py" -u code/analyze_split.py train
"$py" -u code/consistency.py train
