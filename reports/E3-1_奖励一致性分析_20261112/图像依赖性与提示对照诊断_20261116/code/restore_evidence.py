"""Safely restore an immutable archive and check the full byte-level manifest."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import tarfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
j=json.loads(a.metadata.read_text());assert a.archive.stat().st_size==j['archive_bytes'] and sha(a.archive)==j['archive_sha256']
r=a.root/'原始运行记录';r.mkdir(parents=True,exist_ok=True);same=new=0
with tarfile.open(a.archive) as t:
 members=t.getmembers();assert len(members)==j['archive_files'] and len({m.name for m in members})==len(members)
 for m in members:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  data=t.extractfile(m).read();p=r/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists() and m.name!='file_checksums.json':assert p.read_bytes()==data,m.name;same+=1
  else:p.write_bytes(data);new+=1
manifest=json.loads((r/'file_checksums.json').read_text());assert len(manifest)==j['manifest_files']
for n,v in manifest.items():
 p=r/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
j.update(local_archive_and_file_verification='passed',local_manifest_files_verified=len(manifest),existing_files_confirmed_unchanged=same,new_files_restored=new,local_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
a.metadata.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'archive_sha256':j['archive_sha256'],'manifest_files_verified':len(manifest),'existing_unchanged':same,'new':new}))
