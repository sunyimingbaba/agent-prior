#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 E3_END_EPOCH=1795593060
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
wait_for() { while [[ ! -f "$1" ]]; do [[ $(date +%s) -lt $E3_END_EPOCH ]] || exit 75; sleep 3; done; }
wait_for results/resolution_nomination.json
if ! "$cpu" -c 'import json,sys;sys.exit(0 if json.load(open("results/resolution_nomination.json"))["eligible"] else 1)'; then exit 0; fi
wait_for results/prefix_nomination.json
wait_for logs/cross_gpu.done
if "$cpu" -c 'import json,sys;sys.exit(0 if json.load(open("results/prefix_nomination.json"))["eligible"] else 1)'; then
  wait_for logs/prefix_train_gpu.done
  tags=(a_ordinal_R3 b_ordinal_R3)
else
  tags=(a_resolutionbase_R3 a_ordinal_R3 b_resolutionbase_R3 b_ordinal_R3)
fi
if [[ "$1" == gpu ]]; then
  for tag in "${tags[@]}"; do
    IFS=_ read -r seed cond arm <<< "$tag"
    "$gpu" -u code/train_rfpo.py "$seed" "$cond" "$arm"
    "$gpu" -u code/learned_eval.py generate "$tag"
  done
else
  CUDA_VISIBLE_DEVICES='' "$cpu" -u code/learned_eval.py project "${tags[@]}"
fi
touch "logs/resolution_$1.done"
