import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
BASE = Path('/private/AGSE_E1_20261110/legacy')
ASSETS = BASE/'assets'
assert ROOT == Path('/private/AGSE_E3_1_rebuild_20261112/state_parity_20261114')
PLAN = json.loads((ROOT/'plan.json').read_text())
SOURCE = Path(PLAN['source_run_root'])
for name in ['scratch', 'scratch/hf', 'scratch/cache', 'scratch/matplotlib']:
    (ROOT/name).mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(ROOT/'scratch/hf'),
    TMPDIR=str(ROOT/'scratch'), XDG_CACHE_HOME=str(ROOT/'scratch/cache'),
    MPLCONFIGDIR=str(ROOT/'scratch/matplotlib'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
sys.path.insert(0, str(BASE/'CUP'))
sys.path.insert(0, str(BASE/'CUP/agent'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temp, path)


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def setup_torch():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def seed(value):
    import torch
    import numpy as np
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
