#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=2
py=/private/AGSE_E1_20261110/legacy/env/bin/python
until [[ -s outputs/fresh/complete.json || -s outputs/train/complete.json ]]; do sleep 4; done
"$py" -u code/choice.py > logs/choice.log 2>&1
"$py" -u code/gallery.py > logs/gallery.log 2>&1
touch logs/aux_complete
