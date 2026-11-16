"""Seal a new evidence archive only after generation, evaluation and post checks pass."""
import argparse
import datetime
from pathlib import Path
import subprocess
import tarfile
from stage_io import read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();r=Path(__file__).resolve().parent.parent
assert str(r)=='/private/AGSE_E3_1_rebuild_20261112/protocol_prompt_state_validation_20261116'
assert Path(a.name).name==a.name and a.name.endswith('.tar.xz')
for n in ['generation_job_result_v2.json','evaluation_job_result.json','post_job_result.json']:assert read(r/n)['status']=='complete',n
for n in ['server_generation_verification.json','server_evaluation_verification.json','server_descriptive_verification.json','final_source_audit.json']:assert read(r/n)['ok'],n
paths=[p for p in sorted(r.rglob('*')) if p.is_file() and not set(p.relative_to(r).parts)&{'scratch','__pycache__','原始运行记录','归档分卷'} and p.name!='file_checksums.json' and not p.name.endswith(('.pyc','.partial','.tar.gz','.tar.xz'))]
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths};dump(r/'file_checksums.json',manifest)
archive=r.parent/a.name;assert not archive.exists()
with tarfile.open(archive,'w:xz',preset=6) as t:
 for p in paths+[r/'file_checksums.json']:t.add(p,arcname=str(p.relative_to(r)))
generation=read(r/'generation_result_v2.json')
info={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_root':str(r),'archive_name':archive.name,'remote_archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'archive_files':len(paths)+1,'manifest_files':len(manifest),'Agent_optimizer_updates':generation['Agent_optimizer_updates'],'readout_optimizer_updates':0,'readout_conditions':270,'source_rows':79650,'GPU_encodings':70830,'reused_validation_encodings':8820,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),'parent_dual_backup_reference':read(r/'inputs/parent_dual_backup_reference.json')}
dump(r.parent/(archive.name+'.json'),info);print('PROTOCOL_STATE_EVIDENCE_PACKAGED',len(manifest),archive.stat().st_size,flush=True)
