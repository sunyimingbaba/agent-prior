#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
while [ ! -f inputs/nomination_bounded.json ]; do sleep 4; done
if /private/AGSE_E1_20261110/legacy/env/bin/python -c 'import json,sys;sys.exit(0 if json.load(open("inputs/nomination_bounded.json"))["eligible"] else 1)'; then
  while [ ! -f confirm/greedy.done ]; do sleep 4; done
fi
export E3_PANEL_ROOT="$PWD/reward_confirm" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
mkdir -p reward_confirm/{scratch,records,results}
/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python /private/AGSE_E3_1_rebuild_20261112/r3_informative_groups_20261124/../r3_after_tool_trainpanel_20261124/code/sample_panel.py prepare
for phase in first extra; do
  args=(); if [ "$phase" = extra ]; then args=(--extra); fi
  for seed in a b; do
    /private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python /private/AGSE_E3_1_rebuild_20261112/r3_informative_groups_20261124/../r3_after_tool_trainpanel_20261124/code/sample_panel.py "$seed" "${args[@]}"
  done
done
touch reward_confirm/sampled.done
