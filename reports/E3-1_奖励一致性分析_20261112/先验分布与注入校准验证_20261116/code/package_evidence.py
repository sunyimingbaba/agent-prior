import argparse
import datetime
from pathlib import Path
import subprocess
import tarfile
from stage_io import read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--name',required=True);name=p.parse_args().name;r=Path(__file__).resolve().parent.parent;assert str(r).startswith('/private/AGSE_');assert read(r/'post_result.json')['status']=='complete' and read(r/'gpu_retry_result.json')['status']=='complete'
paths=[p for p in sorted(r.rglob('*')) if p.is_file() and not set(p.relative_to(r).parts)&{'scratch','__pycache__'} and p.relative_to(r).parts[0]!='原始运行记录' and str(p.relative_to(r))!='file_checksums.json' and not p.name.endswith('.tar.gz')]
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths};dump(r/'file_checksums.json',manifest);archive=r.parent/name;assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
    for p in paths+[r/'file_checksums.json']:t.add(p,arcname=str(p.relative_to(r)))
j={'archive_name':archive.name,'remote_archive':str(archive),'remote_root':str(r),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'archive_files':len(manifest)+1,'manifest_files':len(manifest),'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_after_run':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),'new_Qwen_generations':0,'optimizer_updates':0,'new_scores':1470,'input_gradient_witnesses':32};dump(r.parent/(name+'.json'),j);print('INJECTION_EVIDENCE_PACKAGED',len(manifest),archive.stat().st_size)
