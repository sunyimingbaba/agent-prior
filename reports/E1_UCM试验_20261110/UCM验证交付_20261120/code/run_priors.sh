#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
for method in ASPF AGSE AGSE_scene; do
    "$py" -u code/priors.py "$method"
done
