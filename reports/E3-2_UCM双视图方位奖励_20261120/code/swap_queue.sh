#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=2
until [[ -f logs/eval_complete ]]; do sleep 10; done
py=/private/AGSE_E1_20261110/legacy/env/bin/python
"$py" -u code/crossed_readout.py > logs/crossed_readout.log 2>&1
until [[ -s results/detail_by_source.csv && -s results/decision.json ]]; do sleep 5; done
"$py" -u code/analyze_swap.py > logs/analyze_swap.log 2>&1
touch logs/swap_complete
