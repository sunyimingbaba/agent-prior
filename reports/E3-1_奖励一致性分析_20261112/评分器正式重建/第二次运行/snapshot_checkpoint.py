#!/usr/bin/env python3
"""Copy a completed-epoch checkpoint to a new recovery snapshot while training continues."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--completed-epochs', type=int, required=True)
args = parser.parse_args()
root = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112/attempt_02')
sys.path.insert(0, '/private/AGSE_E1_20261110/legacy/CUP')
import torch

progress = json.loads((root/'progress.json').read_text(encoding='utf-8'))
assert progress.get('completed_epochs', 0) >= args.completed_epochs
source = root/'run_01/checkpoints/last.ckpt'
out = root/'snapshots'/f'epoch_{args.completed_epochs}'
out.mkdir(parents=True, exist_ok=False)
before = source.stat()
shutil.copy2(source, out/'last.ckpt')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


checksum = sha(out/'last.ckpt')
assert checksum == sha(source)
after = source.stat()
assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
saved = torch.load(out/'last.ckpt', map_location='cpu', weights_only=False, mmap=True)
assert saved['epoch'] == args.completed_epochs
assert saved['global_step'] == args.completed_epochs*32
for name in ['training_events.jsonl', 'batch_losses.jsonl']:
    lines = (root/'run_01'/name).read_text(encoding='utf-8').splitlines(keepends=True)
    selected = [line for line in lines if line.endswith('\n')
                and json.loads(line).get('epoch', -1) < args.completed_epochs]
    (out/name).write_text(''.join(selected), encoding='utf-8')
for name in ['resolved_settings.json', 'preflight_result.json']:
    shutil.copy2(root/name, out/name)
summary = {'checkpoint': str(out/'last.ckpt'), 'bytes': (out/'last.ckpt').stat().st_size,
    'sha256': checksum, 'stored_epoch': saved['epoch'], 'stored_global_step': saved['global_step'],
    'role': 'intermediate recovery backup; final scorer remains the checkpoint after 50 epochs'}
(out/'snapshot_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary))
