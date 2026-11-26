#!/bin/bash
set -euo pipefail
cd /private/AGSE_E3_1_rebuild_20261112/greedy_attainable_20261126
while [[ ! -f ../greedy_margin_20261126/logs/cross.done ]]; do sleep 5; done
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
for s in a b; do for arm in ordinary teacher; do "$py" -Bu code/fit.py "$s" "$arm"; done; done
touch logs/train.done
for s in a b; do for arm in ordinary teacher; do "$py" -Bu code/evaluate.py "$s" "$arm"; "$py" -Bu code/fitcheck.py "$s" "$arm"; done; done
touch logs/eval.done
for s in a b; do "$py" -Bu code/cross.py "$s" teacher; done
touch logs/cross.done
