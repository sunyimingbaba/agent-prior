#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=4 PYTHONUTF8=1 LC_ALL=C.UTF-8
py=/private/AGSE_E1_20261110/legacy/env/bin/python
[[ -s train/decision.json ]]
"$py" -u code/native_reward.py train
until [[ -s samples/rotated/1687/sample7.json ]]; do sleep 5; done
"$py" -u code/score_split.py val
"$py" -u code/analyze_split.py val
"$py" -u code/consistency.py val
"$py" -u code/native_reward.py val
"$py" -u code/sensitivity.py
"$py" -u code/full_val.py
"$py" -u code/rotation_swap.py
"$py" -u code/verify.py
echo VERIFIED
"$py" -u code/full_val_stable.py
