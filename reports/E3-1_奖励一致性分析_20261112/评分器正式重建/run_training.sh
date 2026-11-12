#!/bin/bash
set -u
TASK_DIR=/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112
PYTHONUTF8=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 /private/AGSE_E1_20261110/legacy/env/bin/python -u "$TASK_DIR/train_scorer.py" > "$TASK_DIR/run_01.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "$TASK_DIR/run_01.exit"
exit "$TASK_EXIT"
