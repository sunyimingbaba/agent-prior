#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
score() {
    local arm="$1" step="$2" tag
    tag=$(printf '%s_%02d' "$arm" "$step")
    if [[ -s eval/$tag/complete.json ]]; then return; fi
    if [[ "$arm" != base ]]; then
        local checkpoint
        checkpoint=$(printf 'policy_%s/step_%02d/update.json' "$arm" "$step")
        until [[ -s "$checkpoint" ]]; do sleep 5; done
    fi
    "$py" -u code/evaluate.py "$arm" "$step" > "logs/eval_$tag.log" 2>&1
}
score base 0
for arm in a_separate a_joint b_separate b_joint a_single b_single; do
    for step in 8 16; do score "$arm" "$step"; done
done
touch logs/eval_complete
