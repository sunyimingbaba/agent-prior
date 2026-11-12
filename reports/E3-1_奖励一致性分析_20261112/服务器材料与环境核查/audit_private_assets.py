#!/usr/bin/env python3
"""Audit private assets on CPU; source files are only read."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import torch

ROOT = Path('/private/AGSE_E1_20261110')
ASSETS = ROOT/'legacy/assets'
OUT = Path('/private/AGSE_E3_1_rebuild_20261112/audit_20261112')
torch.set_num_threads(2)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


result = {'root': str(ROOT), 'source_assets_modified': False,
          'mount': subprocess.check_output(['findmnt', '-T', '/private', '-J'], text=True),
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free',
                                           '--format=csv,noheader'], text=True),
          'files': {}, 'arrows': {}, 'prior_groups': {}, 'source_hashes': {}}
for name, relative in {'agse': 'ucm_main_domain_tool.ckpt',
                       'clip': 'models/ViT-L-14.pt',
                       'lora': 'ucm_main_lora/adapter_model.safetensors',
                       'old_feature_cache': 'grpo_val_cache.pt'}.items():
    p = ASSETS/relative
    result['files'][name] = {'path': str(p), 'exists': p.is_file(),
                             'bytes': p.stat().st_size, 'sha256': sha(p)}
    print('HASHED', name, flush=True)
base = ASSETS/'models/Qwen2.5-VL-7B-Instruct'
index = json.loads((base/'model.safetensors.index.json').read_text())
result['qwen'] = {'path': str(base),
    'index_sha256': sha(base/'model.safetensors.index.json'),
    'config_sha256': sha(base/'config.json'),
    'original_config_sha256': sha(base/'config.original.json') if (base/'config.original.json').exists() else None,
    'shards': {name: {'exists': (base/name).is_file(), 'bytes': (base/name).stat().st_size}
               for name in sorted(set(index['weight_map'].values()))}}
for split in ['train', 'val', 'test']:
    p = ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow'
    table = pa.ipc.open_file(str(p)).read_all()
    result['arrows'][split] = {'path': str(p), 'rows': len(table),
                               'sha256': sha(p), 'columns': table.column_names}
    prior_dir = ROOT/'ucm_prior_archive/agent_cache_domain_tool'/p.stem
    names = {x.name for x in prior_dir.glob('*.pt')}
    samples = []
    for i in sorted({0, len(table)//2, len(table)-1}):
        prior = torch.load(prior_dir/f'{i}.pt', map_location='cpu', weights_only=True)
        samples.append({'row': i, 'shape': list(prior.shape), 'dtype': str(prior.dtype),
                         'finite': bool(torch.isfinite(prior).all()),
                         'sha256': sha(prior_dir/f'{i}.pt')})
    result['prior_groups'][split] = {'path': str(prior_dir), 'count': len(names),
        'filenames_complete': names == {f'{i}.pt' for i in range(len(table))}, 'sample_checks': samples}
for relative in ['agent/rollout.py', 'agent/grpo/scorer.py', 'agent/grpo/reward.py',
                 'agent/grpo/train_grpo.py', 'vilt/config.py', 'run_frozen.py',
                 'vilt/modules/vilt_module.py']:
    p = ROOT/'legacy/CUP'/relative
    result['source_hashes'][relative] = sha(p)
misc = ROOT/'legacy/cache/ucm_caption_karpathy_label_train'
result['legacy_debug_cache'] = {'path': str(misc), 'files': sorted(p.name for p in misc.glob('*.pt'))}
for current, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in ['env', 'venv', 'wheels', '__pycache__', '.git'] and not d.startswith('Qwen')]
    if any(f.endswith('.ckpt') for f in files):
        result.setdefault('checkpoint_paths', []).extend(str(Path(current)/f) for f in files if f.endswith('.ckpt'))
(OUT/'private_assets.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'files': result['files'], 'arrows': result['arrows'],
                  'prior_group_counts': {k: v['count'] for k, v in result['prior_groups'].items()},
                  'legacy_debug_cache': result['legacy_debug_cache']}, ensure_ascii=False), flush=True)
