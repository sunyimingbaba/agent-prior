#!/bin/bash
set -eu
cd "$(dirname "$0")/../.."
exec 9>scoring/finish.lock
flock -n 9
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/private/AGSE_E1_20261110/legacy/env/bin/python
until test -f sampling/a_complete.json && test "$(find scores/a -name '*.json' 2>/dev/null | wc -l)" -eq 270; do sleep 5; done
"$py" scoring/code/screen.py a
until test -f sampling/b_eligibility.json; do sleep 5; done
"$py" scoring/code/screen.py lock
"$py" scoring/code/pool_reader.py --seed b --shard 0 --nshards 2 >scoring/b_0.log 2>&1 &
first=$!
"$py" scoring/code/pool_reader.py --seed b --shard 1 --nshards 2 >scoring/b_1.log 2>&1 &
second=$!
wait "$first"
wait "$second"
"$py" scoring/code/screen.py b
