#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
tags=()
for seed in a b; do
  for cond in scene selected; do
    for arm in R2 R3; do tags+=("${seed}_${cond}_${arm}"); done
  done
done
[[ -f results/nomination.json ]]
"$cpu" -c 'import json; assert json.load(open("results/nomination.json"))["selected"] in ("mixed","question")'
if [[ "$1" == gpu ]]; then
  for tag in "${tags[@]}"; do
    IFS=_ read -r seed cond arm <<< "$tag"
    "$gpu" -u code/train_rfpo.py "$seed" "$cond" "$arm"
  done
  touch logs/train.done
  for tag in "${tags[@]}"; do "$gpu" -u code/learned_eval.py generate "$tag"; done
  touch logs/learned_gpu.done
else
  CUDA_VISIBLE_DEVICES='' "$cpu" -u code/learned_eval.py project "${tags[@]}"
  touch logs/learned_cpu.done
fi
