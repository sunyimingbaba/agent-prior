#!/usr/bin/env python3
"""Package a completed cache and its evidence without scratch/cache directories."""
import hashlib
import json
import os
import tarfile
from pathlib import Path

root = Path('/private/AGSE_E3_1_rebuild_20261112/plain_prior_full_20261112')
archive = root.parent/'plain_prior_full_evidence_20261112.tar.gz'
metadata = json.loads((root/'run_01/metadata.json').read_text(encoding='utf-8'))
assert metadata['ok'] and metadata['status'] == 'complete'
assert metadata['completed'] == 1890 and metadata['smoke_anchor_matches'] == 4
assert (root/'run_01.exit').read_text().strip() == '0'

selected = []
for name in ['input_model_manifest.json', 'generate_full_priors.py',
             'run_full_generation.sh', 'monitor_full.py', 'verify_full_priors.py',
             'collect_evidence.py', 'run_01.log', 'run_01.exit']:
    selected.append(root/name)
for path in sorted((root/'run_01').rglob('*')):
    relative = path.relative_to(root/'run_01')
    if relative.parts[0] == 'scratch' or path.name == 'generation.lock':
        continue
    if path.is_file():
        selected.append(path)
assert all(p.is_file() and not p.is_symlink() for p in selected)
files = []
for path in selected:
    files.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
manifest_path = root/'evidence_file_manifest.json'
manifest_path.write_text(json.dumps({'files': files, 'file_count': len(files)},
                                   ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
if archive.exists():
    raise FileExistsError(f'Preserve the existing archive: {archive}')
temporary = archive.with_name(archive.name+'.partial')
with tarfile.open(temporary, 'w:gz', compresslevel=6) as tar:
    for path in [*selected, manifest_path]:
        tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
os.replace(temporary, archive)
result = {'path': str(archive), 'bytes': archive.stat().st_size,
          'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'members': len(files)+1, 'cache_count': 1890}
(root/'archive_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
