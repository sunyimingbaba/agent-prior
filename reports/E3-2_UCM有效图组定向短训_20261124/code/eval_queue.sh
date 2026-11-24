#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
while [ ! -f pilot/train.done ]; do sleep 5; done
export E3_PANEL_ROOT="$PWD"
/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python /private/AGSE_E3_1_rebuild_20261112/r3_after_tool_trainpanel_20261124/code/sample_panel.py prepare
for s in a b; do
  for arm in start regular_R3 selected_R3 focused_R3 focused_R2; do
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python -u code/evaluate.py "$s" "$arm"
  done
done
touch pilot/eval.done
