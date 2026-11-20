#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
run_eval() {
    local arm="$1" step="$2" suffix="$3"
    local run
    run=$(printf '%s_%02d%s' "$arm" "$step" "$suffix")
    if [[ -s "eval/$run/complete.json" ]]; then return; fi
    if [[ "$arm" != base ]]; then
        local checkpoint
        checkpoint=$(printf 'policy_%s/step_%02d/update.json' "$arm" "$step")
        until [[ -s "$checkpoint" ]]; do sleep 5; done
    fi
    if [[ "$suffix" == _force ]]; then
        "$py" -u code/evaluate.py "$arm" "$step" --suppress > "logs/eval_$run.log" 2>&1
    else
        "$py" -u code/evaluate.py "$arm" "$step" > "logs/eval_$run.log" 2>&1
    fi
}
run_eval base 0 ''
for arm in a_off a_on b_off b_on; do
    for step in 6 12 24; do run_eval "$arm" "$step" ''; done
done
for arm in b_off b_on; do run_eval "$arm" 24 _force; done
touch logs/eval_complete
