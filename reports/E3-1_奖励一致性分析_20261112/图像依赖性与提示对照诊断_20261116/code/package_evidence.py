"""Create a new immutable XZ evidence archive after every phase job succeeds."""
import argparse
import datetime
from pathlib import Path
import subprocess
import tarfile
from stage_io import read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();r=Path(__file__).resolve().parent.parent
assert str(r).startswith('/private/AGSE_E3_1_rebuild_20261112/image_dependence_diagnostics_20261116')
assert Path(a.name).name==a.name and a.name.endswith('.tar.xz')
for name in ['generation_job_result.json','evaluation_job_result.json','post_job_result.json']:assert read(r/name)['status']=='complete',name
for name in ['server_verification.json','server_path_state_verification.json','final_source_audit.json']:assert read(r/name)['ok'],name
paths=[p for p in sorted(r.rglob('*')) if p.is_file() and not set(p.relative_to(r).parts)&{'scratch','__pycache__','原始运行记录'} and p.name!='file_checksums.json' and not p.name.endswith(('.pyc','.partial','.tar.gz','.tar.xz'))]
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths};dump(r/'file_checksums.json',manifest)
archive=r.parent/a.name;assert not archive.exists()
with tarfile.open(archive,'w:xz',preset=6) as t:
 for p in paths+[r/'file_checksums.json']:t.add(p,arcname=str(p.relative_to(r)))
info={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_root':str(r),'archive_name':archive.name,'remote_archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'archive_files':len(paths)+1,'manifest_files':len(manifest),'Agent_optimizer_updates':0,'readout_optimizer_updates':0,'logical_states':2184,'new_actual_generations':read(r/'generation_result.json')['new_actual_generations'],'forward_conditions':64,'image_encodings':18880,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),'readout_checkpoint_backup_reference':read(r/'inputs/parent_dual_backup_reference.json')['checkpoints']}
dump(r.parent/(archive.name+'.json'),info);print('DIAGNOSTIC_EVIDENCE_PACKAGED',len(manifest),archive.stat().st_size,flush=True)
