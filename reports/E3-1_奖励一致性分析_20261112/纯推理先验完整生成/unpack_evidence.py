#!/usr/bin/env python3
"""Check the transport checksum, safely extract, and verify every evidence file."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
archive_summary = json.loads(args.summary.read_text(encoding='utf-8'))
checksum = hashlib.sha256(args.archive.read_bytes()).hexdigest()
assert checksum == archive_summary['sha256']
assert args.archive.stat().st_size == archive_summary['bytes']
if args.out.exists():
    raise FileExistsError(f'Use a new extraction directory: {args.out}')
with tarfile.open(args.archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert len(members) == archive_summary['members']
    assert len({m.name for m in members}) == len(members)
    for member in members:
        name = PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts
        assert name.parts and member.name == name.as_posix()
    args.out.mkdir(parents=True)
    for member in members:
        target = args.out/member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source, target.open('xb') as output:
            shutil.copyfileobj(source, output)
file_manifest = json.loads((args.out/'evidence_file_manifest.json').read_text(encoding='utf-8'))
assert len(file_manifest['files']) == file_manifest['file_count']
assert {m.name for m in members} == {'evidence_file_manifest.json'} | {r['path'] for r in file_manifest['files']}
for record in file_manifest['files']:
    path = args.out/record['path']
    assert path.stat().st_size == record['bytes'], record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], record['path']
result = {'verified': True, 'archive_sha256': checksum,
          'archive_bytes': args.archive.stat().st_size,
          'file_count': file_manifest['file_count'],
          'extracted_members': len(members), 'all_file_sizes_and_sha256_match': True}
(args.out.parent/'压缩包核验结果.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
