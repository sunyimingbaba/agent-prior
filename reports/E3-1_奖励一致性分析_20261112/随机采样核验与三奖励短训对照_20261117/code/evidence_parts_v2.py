"""Split/join the evidence archive in48MiB pieces for GitHub, checking size and SHA256."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
a=s.add_parser('split');a.add_argument('archive',type=Path)
a=s.add_parser('join');a.add_argument('index',type=Path);a.add_argument('output',type=Path)
a=p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
if a.action=='split':
 assert a.archive.is_file()
 parts=[]
 with a.archive.open('rb') as f:
  while data:=f.read(48*1024*1024):
   out=a.archive.with_name(a.archive.name+f'.part{len(parts):03d}')
   if out.exists():assert out.read_bytes()==data
   else:out.write_bytes(data)
   parts.append({'name':out.name,'bytes':len(data),'sha256':sha(out)})
 j={'archive_name':a.archive.name,'archive_bytes':a.archive.stat().st_size,'archive_sha256':sha(a.archive),'parts':parts}
 out=a.archive.with_name(a.archive.name+'.parts.json')
 out.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(j,ensure_ascii=False))
else:
 j=json.loads(a.index.read_text());assert not a.output.exists()
 tmp=a.output.with_name(a.output.name+'.joining');assert not tmp.exists()
 with tmp.open('xb') as f:
  for v in j['parts']:
   assert Path(v['name']).name==v['name']
   src=a.index.parent/v['name'];assert src.stat().st_size==v['bytes'] and sha(src)==v['sha256']
   with src.open('rb') as srcf:
    for b in iter(lambda:srcf.read(8388608),b''):f.write(b)
 assert tmp.stat().st_size==j['archive_bytes'] and sha(tmp)==j['archive_sha256']
 tmp.rename(a.output);print('EVIDENCE_JOINED_AND_SHA256_VERIFIED')
