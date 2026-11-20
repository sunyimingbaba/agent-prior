#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
py=/private/AGSE_E1_20261110/legacy/env/bin/python
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=2
until [[ -s choice/fresh/complete.json ]]; do sleep 4; done
"$py" -u code/matched_choice.py > logs/matched_choice.log 2>&1
"$py" -u code/matched_post.py > logs/matched_post.log 2>&1
touch logs/matched_complete
