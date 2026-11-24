#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export E3_PANEL_ROOT="$PWD/guard" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
for phase in first extra; do
  args=()
  if [ "$phase" = extra ]; then args=(--extra); fi
  /private/AGSE_E1_20261110/legacy/env/bin/python -u code/guard_read.py "${args[@]}"
  /private/AGSE_E1_20261110/legacy/env/bin/python -u /private/AGSE_E3_1_rebuild_20261112/r3_informative_groups_20261124/../r3_after_tool_trainpanel_20261124/code/pack.py "${args[@]}"
done
touch guard/scored.done
