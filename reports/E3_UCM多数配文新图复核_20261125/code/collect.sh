#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export E3_PANEL_ROOT="$PWD" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
base=/private/AGSE_E3_1_rebuild_20261112
sampler="$base/r3_after_tool_trainpanel_20261124/code"
reader="$base/paired_answer_preference_20261124/code/guard_read.py"
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
cpu=/private/AGSE_E1_20261110/legacy/env/bin/python
mkdir -p logs scratch records results
case "$1" in
 gpu)
  "$gpu" -u "$sampler/sample_panel.py" prepare
  for arm in a b; do "$gpu" -u "$sampler/sample_panel.py" "$arm"; done
  for arm in a b; do "$gpu" -u "$sampler/sample_panel.py" "$arm" --extra; done
  touch sampled.done
  ;;
 cpu)
  for phase in first extra; do
    args=()
    if [[ "$phase" == extra ]]; then args=(--extra); fi
    "$cpu" -u "$reader" "${args[@]}"
    "$cpu" -u "$sampler/pack.py" "${args[@]}"
  done
  touch scored.done
  ;;
 *) exit 2 ;;
esac
