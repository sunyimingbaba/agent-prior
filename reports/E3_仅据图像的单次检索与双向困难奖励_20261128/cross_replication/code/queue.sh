#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/lock
flock -n 9
until test -f ../scene_evidence/done && test -s inputs/gate.json; do sleep 6; done
py=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 PYTORCH_ALLOC_CONF=expandable_segments:True
branch=$("$py" -c 'import json;print(json.load(open("inputs/gate.json"))["branch"])')
if [[ "$branch" == "skip" ]]; then touch skipped done; exit 0; fi
"$py" -Bu code/generate.py verify >logs/verify.log 2>&1
for tag in $("$py" -c 'import json;print(" ".join(json.load(open("inputs/gate.json"))["tags"]))'); do
 "$py" -Bu code/generate.py "$tag" >"logs/$tag.log" 2>&1
done
touch done
