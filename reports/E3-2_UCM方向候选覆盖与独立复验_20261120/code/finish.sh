#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=2
until [[ -f logs/aux_complete && -s outputs/train/complete.json && -s outputs/fresh/complete.json ]]; do sleep 4; done
py=/private/AGSE_E1_20261110/legacy/env/bin/python
"$py" -u code/analyze.py > logs/analysis.log 2>&1
"$py" -u code/verify.py ../paired_views_20261120 /private/AGSE_E1_20261110/legacy/assets/ucm > logs/verify.log 2>&1
touch logs/analysis_complete
