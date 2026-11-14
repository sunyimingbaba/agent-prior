"""Create a checksummed evidence archive, leaving the model archive separate."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
r=Path(__file__).resolve().parent.parent;assert str(r).startswith('/private/')
assert json.loads((r/'post_result.json').read_text())['status']=='complete'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

paths=[p for p in sorted(r.rglob('*')) if p.is_file() and p.name!='file_checksums.json'
       and not any(x in p.relative_to(r).parts for x in ['scratch','__pycache__','checkpoint'])]
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}
(r/'file_checksums.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
archive=r.parent/a.name;assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
    for path in paths+[r/'file_checksums.json']:t.add(path,arcname=str(path.relative_to(r)))
backup={'archive_name':archive.name,'remote_archive':str(archive),'remote_root':str(r),
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'archive_files':len(paths)+1,'manifest_files':len(paths),
        'excluded':['scratch','__pycache__','training/*/checkpoint (separate model archive)'],
        'gpu_after_run':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r.parent/(a.name+'.json')).write_text(json.dumps(backup,ensure_ascii=False,indent=2)+'\n');print(json.dumps(backup))
