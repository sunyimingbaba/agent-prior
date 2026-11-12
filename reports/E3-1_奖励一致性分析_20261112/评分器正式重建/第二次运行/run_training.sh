#!/bin/bash
set -u -C
TASK_DIR=/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112/attempt_02
if [[ -e "$TASK_DIR/run_01.log" || -e "$TASK_DIR/run_01.exit" || -d "$TASK_DIR/run_01/logs" ]]; then
  printf '%s\n' 'Existing run evidence found; preserve it and use an audited resume entry.' >&2
  exit 73
fi
PYTHONUTF8=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 /private/AGSE_E1_20261110/legacy/env/bin/python -u "$TASK_DIR/train_scorer.py" > "$TASK_DIR/run_01.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "$TASK_DIR/run_01.exit"
exit "$TASK_EXIT"
