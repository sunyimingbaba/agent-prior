"""Safely restore a downloaded evidence archive and verify every file."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

backup=json.loads(a.metadata.read_text())
assert a.archive.stat().st_size==backup['archive_bytes'] and sha(a.archive)==backup['archive_sha256']
out=a.root/'原始运行记录';out.mkdir(exist_ok=True);unchanged=added=0
with tarfile.open(a.archive) as t:
    members=t.getmembers();assert len(members)==backup['archive_files']
    assert len({m.name for m in members})==len(members)
    for member in members:
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        data=t.extractfile(member).read();path=out/member.name
        if path.exists() and member.name!='file_checksums.json':
            assert path.read_bytes()==data,'Existing evidence changed: '+member.name;unchanged+=1
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);added+=1
manifest=json.loads((out/'file_checksums.json').read_text());assert len(manifest)==backup['manifest_files']
for name,info in manifest.items():
    path=out/name;assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],name
backup.update(local_copy_verification='passed',local_file_hashes_verified=len(manifest),
              existing_files_confirmed_unchanged=unchanged,new_files_restored=added,
              verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
a.metadata.write_text(json.dumps(backup,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(backup,ensure_ascii=False))
