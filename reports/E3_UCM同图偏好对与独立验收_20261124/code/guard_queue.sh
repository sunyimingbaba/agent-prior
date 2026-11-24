#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export E3_PANEL_ROOT="$PWD/guard" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
mkdir -p guard/{records,scratch,results}
for phase in first extra; do
  args=()
  if [ "$phase" = extra ]; then args=(--extra); fi
  for seed in a b; do
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python /private/AGSE_E3_1_rebuild_20261112/r3_informative_groups_20261124/../r3_after_tool_trainpanel_20261124/code/sample_panel.py "$seed" "${args[@]}"
  done
done
touch guard/sampled.done
