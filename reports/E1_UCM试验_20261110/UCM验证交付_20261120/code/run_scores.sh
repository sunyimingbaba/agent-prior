#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 LC_ALL=C.UTF-8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py=/private/AGSE_E1_20261110/legacy/env/bin/python
for method in CLIP CUP_CCB AGSE_archived ASPF AGSE AGSE_scene; do
    "$py" -u code/score.py "$method"
done
