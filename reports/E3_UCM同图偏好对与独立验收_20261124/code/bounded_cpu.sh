#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
while [ ! -f guard/scored.done ]; do sleep 3; done
for s in a b; do
 while [ ! -f "records/${s}_bound_source.pt" ]; do sleep 3; done
 /private/AGSE_E1_20261110/legacy/env/bin/python -u code/bound_reader.py ref "$s"
done
while [ ! -f pilot/bound_train.done ]; do sleep 5; done
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/read_pair.py a_bounded b_bounded
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/pack_stage.py bounded a_bounded b_bounded
touch pilot/bound_scored.done
