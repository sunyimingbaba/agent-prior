"""Verify and archive a user-paused generation queue without changing recorded runs."""
from pathlib import Path
import datetime
import json
import tarfile
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
assert str(r)=='/private/AGSE_E3_1_rebuild_20261112/protocol_prompt_state_validation_20261116'
pause=read(r/'pause_20261116.json');assert pause['status']=='paused_by_user' and not pause['active_phase_jobs']
for n in ['generation_code_hashes.json','generation_code_hashes_v2.json','evaluation_code_hashes.json','post_code_hashes.json','delivery_code_hashes.json']:
 for path,h in read(r/n).items():assert sha(r/path)==h,(n,path)
records=[json.loads(s) for s in (r/'generation_v2/records.jsonl').read_text().splitlines()];assert len(records)==pause['completed_records'] and len({(x['group'],x['key']) for x in records})==len(records)
for x in records:
 for kind in ['trace','tensor']:assert sha(r/x[kind+'_file'])==x[kind+'_sha256'],x['key']
for n,v in read(r/'v1_preserved_inventory.json')['files'].items():assert (r/n).stat().st_size==v['bytes'] and sha(r/n)==v['sha256'],n
reference=read(r/'inputs/balanced_checkpoint_reference.json');checkpoint=Path(read(r/'protocol_plan.json')['policy']['checkpoint'])
for n,v in reference['files'].items():assert sha(checkpoint/n)==v['sha256'],n
traced={x['trace_file'] for x in records};tensor={x['tensor_file'] for x in records};orphans=[str(p.relative_to(r)) for p in (r/'generation_v2').rglob('*') if p.is_file() and p.suffix in ['.pt','.json'] and not p.name.startswith('anchor_') and str(p.relative_to(r)) not in traced|tensor]
out={'ok':True,'scope':'partial paused generation, byte-level preservation; final generation/evaluation checks remain pending','completed_records':len(records),'record_trace_and_tensor_hashes_verified':2*len(records),'original_policy_checkpoint_files_verified':len(reference['files']),'V1_preserved_files_verified':len(read(r/'v1_preserved_inventory.json')['files']),'orphan_files_preserved':orphans,'Agent_optimizer_updates':0,'readout_optimizer_updates':0,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};dump(r/'paused_partial_verification.json',out)
paths=[p for p in sorted(r.rglob('*')) if p.is_file() and not set(p.relative_to(r).parts)&{'scratch','__pycache__','原始运行记录','归档分卷'} and p.name!='file_checksums.json' and not p.name.endswith(('.pyc','.partial','.tar.gz','.tar.xz'))]
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths};dump(r/'file_checksums.json',manifest)
archive=r.parent/'protocol_state_pause_20261116.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for p in paths+[r/'file_checksums.json']:t.add(p,arcname=str(p.relative_to(r)))
info={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'paused_by_user','remote_root':str(r),'archive_name':archive.name,'remote_archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'archive_files':len(paths)+1,'manifest_files':len(manifest),'completed_records':len(records),'planned_records':672,'Agent_optimizer_updates':0,'readout_optimizer_updates':0,'partial_verification':out}
dump(r.parent/(archive.name+'.json'),info);print(json.dumps(info,ensure_ascii=False),flush=True)
