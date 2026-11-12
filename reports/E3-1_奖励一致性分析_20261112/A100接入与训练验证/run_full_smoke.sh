#!/bin/bash
set -u
AUDIT_DIR=/private/AGSE_E3_1_rebuild_20261112/a100_audit_20261112
/private/AGSE_E1_20261110/legacy/env/bin/python "$AUDIT_DIR/full_training_smoke.py" --batch-size 32 > "$AUDIT_DIR/full_training_smoke.log" 2>&1
SMOKE_EXIT=$?
printf '%s\n' "$SMOKE_EXIT" > "$AUDIT_DIR/full_training_smoke.exit"
exit "$SMOKE_EXIT"
