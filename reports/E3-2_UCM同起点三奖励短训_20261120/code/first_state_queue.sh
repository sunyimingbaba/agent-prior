#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
# Start after a second evaluated arm has released its decoder memory.
until [[ -s eval/hard/scored.json || -s eval/caption/scored.json ]]; do sleep 10; done
for arm in base hard caption old; do
    until [[ -s eval/$arm/scored.json ]]; do sleep 10; done
    if [[ ! -s first_state/$arm/summary.json ]]; then
        "$py" -u code/first_state.py "$arm" > "logs/first_state_$arm.log" 2>&1
    fi
done
touch logs/first_state_complete
