#!/usr/bin/env python3
"""Package the stopped run's evidence, preserving checkpoint binaries separately."""
import hashlib
import json
import os
import tarfile
from pathlib import Path

parent = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112')
active = parent/'attempt_02'
status = json.loads((active/'pause_status.json').read_text(encoding='utf-8'))
assert status['status'] == 'user_paused' and status['trainer_process_stopped']
archive = parent.parent/'scorer_pause_evidence_20261112.tar.gz'
assert not archive.exists()
selected = []
for path in sorted(parent.rglob('*')):
    relative = path.relative_to(parent)
    if any(part in ['scratch', 'checkpoints', '__pycache__'] for part in relative.parts):
        continue
    if path.name.endswith(('.lock', '.partial', '.ckpt')):
        continue
    if path.name == 'pause_evidence_file_manifest.json':
        continue
    if path.is_file():
        assert not path.is_symlink()
        selected.append(path)
files = [{'path': path.relative_to(parent).as_posix(), 'bytes': path.stat().st_size,
          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in selected]
manifest_path = parent/'pause_evidence_file_manifest.json'
manifest_path.write_text(json.dumps({'files': files, 'file_count': len(files),
    'checkpoint_binaries': 'backed up separately with registered SHA-256'},
    ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
temporary = archive.with_name(archive.name+'.partial')
with tarfile.open(temporary, 'w:gz', compresslevel=6) as tar:
    for path in selected:
        tar.add(path, arcname=path.relative_to(parent).as_posix(), recursive=False)
    tar.add(manifest_path, arcname='evidence_file_manifest.json', recursive=False)
os.replace(temporary, archive)
summary = {'path': str(archive), 'bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'members': len(files)+1,
    'status': 'user_paused', 'preferred_resume_completed_epochs': 30}
(parent/'pause_archive_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary))
