#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
while [[ ! -f pilot/training.done ]]; do sleep 5; done
for group in 'a start' 'b start' 'a R2' 'a R3' 'b R2' 'b R3'; do
    read -r seed reward <<< "$group"
    "$py" -u code/pilot_eval.py "$seed" "$reward" > "pilot/${seed}_${reward}.eval.log" 2>&1
done
touch pilot/eval.done
