"""Capture actual run markers, per-arm coverage, and checksummed process logs on the private mount."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def count(path):return sum(1 for p in path.glob('*.pt') if p.is_file())
def main():
    logs=ROOT/'logs';records=ROOT/'records';result=ROOT/'results';result.mkdir(exist_ok=True)
    files=sorted(p for p in logs.iterdir() if p.is_file() and p.suffix in ('.log','.done'))
    archive=records/'execution_logs.tar.gz'
    with tarfile.open(archive.with_suffix('.part'),'w:gz',compresslevel=5) as tar:
        for p in files:tar.add(p,arcname=p.name,recursive=False)
    archive.with_suffix('.part').replace(archive)
    dirs={}
    for base in ['first','second','prefix/second','prefix/third','route','cross','stochastic','learned']:
        for tag in sorted((records/base).glob('*')):
            if not tag.is_dir():continue
            groups={k:count(tag/k) for k in ('raw','features','primary','official') if (tag/k).is_dir()}
            dirs[base+'/'+tag.name]=dict(file_counts=groups,sealed=[p.name for p in sorted(tag.glob('*.json'))])
    pilots={}
    for p in sorted((ROOT/'pilot').glob('*')):
        if not p.is_dir():continue
        pilots[p.name]={k:json.loads((p/k).read_bytes()) for k in ('complete.json','window_end.json') if (p/k).exists()}
        if (p/'resume.pt').exists():pilots[p.name]['resume_sha256']=digest(p/'resume.pt')
    data=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scheduled_end_utc='2026-11-25T07:55:22+00:00',
              server_directory=str(ROOT),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip(),
              cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),markers=[p.name for p in files if p.suffix=='.done'],coverage=dirs,pilots=pilots,
              logs_sha256=digest(archive),bytes=archive.stat().st_size,logs={p.name:digest(p) for p in files})
    target=result/'window.json';temp=target.with_suffix('.part');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(target)
if __name__=='__main__':main()
