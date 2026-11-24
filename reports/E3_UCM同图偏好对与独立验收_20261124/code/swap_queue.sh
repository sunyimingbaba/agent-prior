#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/swap_reader.py fresh
while [ ! -f reward_confirm/anchor.done ]; do sleep 5; done
/private/AGSE_E1_20261110/legacy/env/bin/python -u code/swap_reader.py independent
touch reward_confirm/swap.done
