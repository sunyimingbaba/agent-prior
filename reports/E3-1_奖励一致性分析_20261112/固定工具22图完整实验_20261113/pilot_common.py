"""Private paths and recording helpers for the bounded E3-1 pilot."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/private/AGSE_E1_20261110/legacy')
ASSETS = BASE/'assets'
REBUILD = Path('/private/AGSE_E3_1_rebuild_20261112')
assert ROOT.parent == REBUILD and ROOT.name.startswith('fixed_tool_22_20261113')
PLAN = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
for relative in ['scratch', 'scratch/hf', 'scratch/cache', 'scratch/matplotlib']:
    (ROOT/relative).mkdir(parents=True, exist_ok=True)
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HOME': str(ROOT/'scratch/hf'), 'TMPDIR': str(ROOT/'scratch'),
    'MPLCONFIGDIR': str(ROOT/'scratch/matplotlib'), 'XDG_CACHE_HOME': str(ROOT/'scratch/cache'),
    'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
sys.path.insert(0, str(BASE/'CUP'))
sys.path.insert(0, str(BASE/'CUP/agent'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def setup_torch():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def seed(value):
    import random
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

N_IMAGES = len(PLAN["images"])
N_CANDIDATES = N_IMAGES * PLAN["group_size"]
N_GENERATION = N_CANDIDATES + 2*N_IMAGES
N_SCORING = N_GENERATION + N_IMAGES
