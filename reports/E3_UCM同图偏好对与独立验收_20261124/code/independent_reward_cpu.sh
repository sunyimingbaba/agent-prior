#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
while [ ! -d reward_confirm/records/a ]; do sleep 3; done
if /private/AGSE_E1_20261110/legacy/env/bin/python -c 'import json,sys;sys.exit(0 if json.load(open("inputs/nomination_bounded.json"))["eligible"] else 1)'; then
  while [ ! -f confirm/scored.done ]; do sleep 4; done
fi
export E3_PANEL_ROOT="$PWD/reward_confirm" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
for phase in first extra; do
  args=()
  if [ "$phase" = extra ]; then args=(--extra); fi
  /private/AGSE_E1_20261110/legacy/env/bin/python -u code/guard_read.py "${args[@]}"
  /private/AGSE_E1_20261110/legacy/env/bin/python -u /private/AGSE_E3_1_rebuild_20261112/r3_informative_groups_20261124/../r3_after_tool_trainpanel_20261124/code/pack.py "${args[@]}"
done
touch reward_confirm/scored.done
