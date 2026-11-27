#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
gpu=/private/AGSE_E1_20261110/e1_prior_compat_20261111/venv/bin/python
"$gpu" - <<'PY'
import hashlib,json
from pathlib import Path
receipt=Path('inputs/pilot/decision.json')
def checksum(f):
    with open(f,'rb') as infile:return hashlib.file_digest(infile,'sha256').hexdigest()
assert checksum(receipt)=='2b09e3339c7f16a1b7727c9bddebe9c4a22c34c179fea2da23f834b01d94c2fc'
plan=json.loads(receipt.read_text())
assert all(checksum(f)==h for f,h in plan['lock'].items())
assert json.load(open('results/在线评分核对.json'))['groups']==[26,26]
for name in ('a_ordinary','a_old'):
    dest=Path('models')/name
    proof=json.loads((dest/'complete.json').read_text())
    assert proof['lock']==checksum(receipt) and proof['initial']==plan['starts']['a']
    assert proof['final']==checksum(dest/'adapter/adapter_model.safetensors')
    assert proof['scores']==checksum(dest/'steps.json') and proof['steps']==13
PY
for seed in a b; do
 for arm in ordinary old old3; do
  if [[ "$seed" = a && "$arm" != old3 ]]; then continue; fi
  "$gpu" -Bu code/train.py "$seed" "$arm" >"logs/train_${seed}_${arm}.log" 2>&1
 done
done
touch logs/train.done
