#!/usr/bin/env python3
"""Package verified resumed-run evidence and its previously verified source archive."""
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

root = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_resume_20261113')
result = json.loads((root/'training_result.json').read_text(encoding='utf-8'))
metrics = json.loads((root/'training_metrics_verification.json').read_text(encoding='utf-8'))
registry = json.loads((root/'frozen_scorer_registry.json').read_text(encoding='utf-8'))
assert result['ok'] and metrics['verified'] and registry['status'] == 'frozen'
assert result['optimizer_steps'] == 1600 and result['resumed_epochs'] == 20
assert (root/'run_01.exit').read_text().strip() == '0'
archive = root.parent/'scorer_resume_evidence_20261113.tar.gz'
assert not archive.exists()
prefix = root.parent/'scorer_pause_evidence_20261112.tar.gz'
assert hashlib.sha256(prefix.read_bytes()).hexdigest() == 'd66f3366b2d953b1e3527547a0948ed0ac6434ded87fc7a0d42f2d231289763f'
(root/'source_prefix').mkdir(exist_ok=True)
shutil.copy2(prefix, root/'source_prefix'/prefix.name)
selected = []
for path in sorted(root.rglob('*')):
    relative = path.relative_to(root)
    if any(part in ['scratch', 'checkpoints', '__pycache__'] for part in relative.parts):
        continue
    if path.name.endswith(('.lock', '.partial', '.ckpt')):
        continue
    if path.is_file():
        assert not path.is_symlink()
        selected.append(path)
files = [{'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in selected]
manifest = root/'evidence_file_manifest.json'
manifest.write_text(json.dumps({'files': files, 'file_count': len(files),
    'checkpoint_binaries': 'registered and copied independently'}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
temporary = archive.with_name(archive.name+'.partial')
with tarfile.open(temporary, 'w:gz', compresslevel=6) as tar:
    for path in [*selected, manifest]:
        tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
os.replace(temporary, archive)
summary = {'path': str(archive), 'bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'members': len(files)+1,
    'completed_epochs': 50, 'checkpoints': result['checkpoints']}
(root/'archive_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary))
