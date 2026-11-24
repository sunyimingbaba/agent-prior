#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
while [ ! -f guard/scored.done ] || [ ! -f confirm/tools.json ]; do sleep 4; done
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/confirm_jobs.py read
arm=$(/private/AGSE_E1_20261110/legacy/env/bin/python -c 'import json,os;print(json.load(open("inputs/"+os.environ.get("E3_NOMINATION","nomination.json")))["arm"])')
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/pack_stage.py confirm a_start "a_$arm" b_start "b_$arm"
touch confirm/scored.done
