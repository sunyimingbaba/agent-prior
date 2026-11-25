#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 E3_END_EPOCH=1795593060
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
tags=(a_locked_R3 a_free_R3 b_locked_R3 b_free_R3 a_locked_R2 a_free_R2 b_locked_R2 b_free_R2)
while [[ ! -f results/prefix_nomination.json ]]; do
  [[ $(date +%s) -lt $E3_END_EPOCH ]] || exit 75
  sleep 3
done
if ! "$cpu" -c 'import json; assert json.load(open("results/prefix_nomination.json"))["eligible"]'; then
  touch logs/cross_allowed
  exit 0
fi
while [[ ! -f logs/replay.done ]]; do
  [[ $(date +%s) -lt $E3_END_EPOCH ]] || exit 75
  sleep 3
done
if [[ "$1" == gpu ]]; then
  for tag in "${tags[@]}"; do
    IFS=_ read -r seed cond arm <<< "$tag"
    "$gpu" -u code/train_rfpo.py "$seed" "$cond" "$arm"
    "$gpu" -u code/learned_eval.py generate "$tag"
  done
  touch logs/prefix_train_gpu.done logs/cross_allowed
else
  CUDA_VISIBLE_DEVICES='' "$cpu" -u code/learned_eval.py project "${tags[@]}"
  touch logs/prefix_train_cpu.done
fi
