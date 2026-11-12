#!/bin/bash
set -u
TASK_DIR=/private/AGSE_E3_1_rebuild_20261112/plain_prior_full_20261112
PYTHONUTF8=1 /private/AGSE_E1_20261110/legacy/env/bin/python "$TASK_DIR/generate_full_priors.py" --manifest "$TASK_DIR/input_model_manifest.json" --out "$TASK_DIR/run_01" > "$TASK_DIR/run_01.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "$TASK_DIR/run_01.exit"
exit "$TASK_EXIT"
