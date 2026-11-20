#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
for arm in base old caption hard; do
    if [[ "$arm" != base ]]; then
        until [[ -s policy_$arm/update.json ]]; do sleep 10; done
    fi
    for part in native generic detail; do
        if [[ ! -s eval/$arm/$part/complete.json ]]; then
            "$py" -u code/deploy.py "$arm" "$part" >"logs/eval_${arm}_${part}.log" 2>&1
        fi
    done
    if [[ ! -s eval/$arm/scored.json ]]; then
        "$py" -u code/score_eval.py "$arm" >"logs/score_${arm}.log" 2>&1
    fi
done
"$py" code/analysis.py >logs/analysis.log 2>&1
"$py" code/check_training.py >logs/training_audit.log 2>&1
touch logs/stage_complete
