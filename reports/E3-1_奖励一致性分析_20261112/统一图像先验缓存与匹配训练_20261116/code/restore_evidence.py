"""Verify both archive hashes, safely restore their union and check every file."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
j=json.loads(a.metadata.read_text());out=a.root/'原始运行记录';out.mkdir(parents=True,exist_ok=True);unchanged=added=0
for path,key in [(a.evidence,'evidence'),(a.checkpoints,'checkpoints')]:
    info=j[key];assert path.stat().st_size==info['archive_bytes'] and sha(path)==info['archive_sha256']
    with tarfile.open(path) as t:
        ms=t.getmembers();assert len(ms)==info['archive_files'] and len({m.name for m in ms})==len(ms)
        for m in ms:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            data=t.extractfile(m).read();p=out/m.name
            if p.exists() and m.name!='file_checksums.json':
                assert p.read_bytes()==data,'Existing evidence changed: '+m.name;unchanged+=1
            else:
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);added+=1
manifest=json.loads((out/'file_checksums.json').read_text());assert len(manifest)==j['manifest_files']
for n,v in manifest.items():
    p=out/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
j.update(local_archive_and_file_verification='passed',local_manifest_files_verified=len(manifest),
    existing_files_confirmed_unchanged=unchanged,new_files_restored=added,
    local_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
a.metadata.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print(json.dumps(j,ensure_ascii=False))
