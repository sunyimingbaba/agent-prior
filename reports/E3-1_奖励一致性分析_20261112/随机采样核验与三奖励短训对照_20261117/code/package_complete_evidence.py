"""Byte-complete artifact archive: exclude only root checksum itself, retain nested same-name manifests."""
from pathlib import Path
import hashlib
import json
import tarfile
import sys
r=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();assert r.is_dir() and not out.exists() and not out.is_relative_to(r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
files=[]
for p in r.rglob('*'):
 rel=p.relative_to(r)
 if not p.is_file() or any(k in rel.parts for k in ['scratch','__pycache__']):continue
 if len(rel.parts)==1 and (p.name=='file_checksums.json' or p.name.endswith('.tar.gz')):continue
 assert not p.is_symlink();files.append(p)
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)};mp=r/'file_checksums.json';assert not mp.exists();mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');out.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(out,'w:gz',compresslevel=1) as t:
 for p in sorted(files)+[mp]:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
seen=set()
with tarfile.open(out,'r:gz') as t:
 for m in t:
  assert m.isfile() and m.name not in seen;seen.add(m.name);f=t.extractfile(m);h=hashlib.sha256();n=0
  for data in iter(lambda:f.read(8388608),b''):h.update(data);n+=len(data)
  expected=manifest[m.name] if m.name!='file_checksums.json' else {'bytes':mp.stat().st_size,'sha256':sha(mp)};assert h.hexdigest()==expected['sha256'] and n==expected['bytes'],m.name
assert seen==set(manifest)|{'file_checksums.json'};result={'ok':True,'archive':str(out),'archive_bytes':out.stat().st_size,'archive_sha256':sha(out),'files_in_root_manifest':len(manifest),'archive_members':len(seen),'root_manifest_sha256':sha(mp),'all_archive_member_bytes_independently_rehashed':True,'nested_same_name_checksum_manifests_retained':sorted(n for n in manifest if Path(n).name=='file_checksums.json'),'excluded':'scratch, __pycache__, root-relative existing tar.gz archives; root checksum manifest excluded from self hash only'};out.with_suffix('.backup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
