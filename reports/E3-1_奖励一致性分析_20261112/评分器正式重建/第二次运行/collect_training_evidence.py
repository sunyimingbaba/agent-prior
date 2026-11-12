#!/usr/bin/env python3
"""Archive finished training evidence; model checkpoint binaries are backed up separately."""
import hashlib
import json
import os
import tarfile
from pathlib import Path

parent = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112')
active = parent/'attempt_02'
archive = parent.parent/'scorer_training_evidence_20261112.tar.gz'
if archive.exists():
    raise FileExistsError(f'Preserve the existing archive: {archive}')
result = json.loads((active/'training_result.json').read_text(encoding='utf-8'))
registry = json.loads((active/'frozen_scorer_registry.json').read_text(encoding='utf-8'))
assert result['ok'] and result['status'] == 'complete' and result['completed_epochs'] == 50
assert registry['status'] == 'frozen' and registry['inference_finite']
assert registry['selected_checkpoint']['sha256'] == result['checkpoints']['last']['sha256']
assert (active/'run_01.exit').read_text().strip() == '0'
selected = []
for path in sorted(parent.rglob('*')):
    relative = path.relative_to(parent)
    if any(part in ['scratch', 'checkpoints', '__pycache__'] for part in relative.parts):
        continue
    if path.name.endswith(('.lock', '.partial', '.ckpt')):
        continue
    if path.is_file():
        assert not path.is_symlink(), str(path)
        selected.append(path)
files = [{'path': path.relative_to(parent).as_posix(), 'bytes': path.stat().st_size,
          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in selected]
file_manifest = parent/'evidence_file_manifest.json'
file_manifest.write_text(json.dumps({'files': files, 'file_count': len(files),
    'checkpoint_binaries': 'excluded; registered and copied independently'},
    ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
temporary = archive.with_name(archive.name+'.partial')
with tarfile.open(temporary, 'w:gz', compresslevel=6) as tar:
    for path in [*selected, file_manifest]:
        tar.add(path, arcname=path.relative_to(parent).as_posix(), recursive=False)
os.replace(temporary, archive)
summary = {'path': str(archive), 'bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'members': len(files)+1,
    'completed_epochs': 50, 'checkpoints': result['checkpoints']}
(parent/'archive_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary))
