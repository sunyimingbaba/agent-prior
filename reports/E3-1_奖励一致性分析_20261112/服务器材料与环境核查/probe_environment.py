#!/usr/bin/env python3
"""Inspect installed runtimes and run small CPU/GPU autograd probes, without updates."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not str(args.output.resolve()).startswith('/private/AGSE_E3_1_rebuild_20261112/'):
    raise ValueError('Audit output must be in the new private E3-1 directory')
scratch = args.output.parent / ('scratch_' + args.label)
scratch.mkdir(parents=True, exist_ok=True)
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'HF_HOME': str(scratch/'hf'), 'TMPDIR': str(scratch),
                   'MPLCONFIGDIR': str(scratch/'matplotlib'),
                   'XDG_CACHE_HOME': str(scratch/'cache'),
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
packages = ['torch', 'torchvision', 'pytorch-lightning', 'torchmetrics',
            'transformers', 'peft', 'accelerate', 'pyarrow', 'numpy',
            'pillow', 'timm', 'torch-ema', 'sacred', 'einops', 'scipy',
            'scikit-learn', 'ftfy', 'regex', 'safetensors', 'tokenizers']
result = {'label': args.label, 'python': platform.python_version(),
          'executable': sys.executable, 'prefix': sys.prefix,
          'packages': {}, 'imports': {}, 'autograd': {},
          'full_model_forward_backward_tested': False,
          'packages_modified': False, 'formal_training_started': False}
for name in packages:
    try:
        result['packages'][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        result['packages'][name] = None
sys.path.insert(0, '/private/AGSE_E1_20261110/legacy/CUP')
for name in ['torch', 'torchvision', 'pyarrow', 'transformers', 'peft',
             'timm', 'torch_ema', 'sacred', 'pytorch_lightning',
             'vilt.modules.vilt_module']:
    try:
        module = importlib.import_module(name)
        result['imports'][name] = {'ok': True}
    except Exception as error:
        result['imports'][name] = {'ok': False,
            'error': f'{type(error).__name__}: {error}',
            'traceback': traceback.format_exc()}
if result['imports']['torch']['ok']:
    import torch
    torch.set_num_threads(2)
    result['cuda'] = {'available': torch.cuda.is_available(),
                       'build_version': torch.version.cuda,
                       'supported_archs': torch.cuda.get_arch_list()}
    if torch.cuda.is_available():
        result['cuda'].update({'name': torch.cuda.get_device_name(0),
                               'capability': list(torch.cuda.get_device_capability(0))})
    for device, dtype_name in [('cpu', 'float32'), ('cuda', 'float32'), ('cuda', 'bfloat16')]:
        key = device + '_' + dtype_name
        try:
            dtype = getattr(torch, dtype_name)
            a = torch.randn((16, 16), dtype=dtype, device=device, requires_grad=True)
            b = torch.randn((16, 16), dtype=dtype, device=device, requires_grad=True)
            loss = (a @ b).float().square().mean()
            loss.backward()
            finite = bool(torch.isfinite(loss).item() and torch.isfinite(a.grad).all().item()
                          and torch.isfinite(b.grad).all().item())
            if device == 'cuda':
                torch.cuda.synchronize()
            result['autograd'][key] = {'ok': finite, 'loss': float(loss.detach().cpu())}
            del a, b, loss
        except Exception as error:
            result['autograd'][key] = {'ok': False,
                'error': f'{type(error).__name__}: {error}'}
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
result['pip_check'] = {'exit_code': check.returncode, 'stdout': check.stdout, 'stderr': check.stderr}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'label': args.label, 'packages': result['packages'],
                  'imports': {k: {a: b for a, b in v.items() if a != 'traceback'}
                              for k, v in result['imports'].items()},
                  'cuda': result.get('cuda'), 'autograd': result['autograd'],
                  'pip_check_exit_code': check.returncode}, ensure_ascii=False), flush=True)
