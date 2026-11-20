#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OPENBLAS_NUM_THREADS=2
py=/private/AGSE_E1_20261110/legacy/env/bin/python
prior=/private/AGSE_E3_1_rebuild_20261112/matched_policy_20261120
until [[ -f logs/train_complete ]]; do sleep 5; done
"$py" -u code/audit_training.py > logs/audit_training.log 2>&1
until [[ -f logs/replicate_complete ]]; do sleep 5; done
"$py" -u code/analyze.py > logs/analyze.log 2>&1
"$py" -u code/crosscheck.py > logs/crosscheck.log 2>&1
"$py" -u code/check_deployment.py "$prior" > logs/deployment_audit.log 2>&1
"$py" -u code/reward_effect.py > logs/format_credit.log 2>&1
touch logs/post_complete
