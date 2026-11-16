"""Split model checkpoints from reproducible evidence; never replace archives."""
import argparse
import datetime
from pathlib import Path
import subprocess
import tarfile
from stage_io import read,dump,sha

p=argparse.ArgumentParser();p.add_argument('--evidence-name',required=True);args=p.parse_args()
r=Path(__file__).resolve().parent.parent
assert str(r).startswith('/private/AGSE_E3_1_rebuild_20261112/image_only_matched_readout_20261116')
assert read(r/'matched_pipeline_result_v2.json')['status']=='complete'
assert read(r/'post_result_v2.json')['status']=='complete' and read(r/'final_source_audit.json')['ok']
checkpoint_paths=sorted(r.glob('training/*/checkpoints/*.ckpt'))
assert len(checkpoint_paths)==8
cpmanifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in checkpoint_paths}
cpmanifest_path=r/'checkpoint_checksums.json'
if cpmanifest_path.exists():
    assert read(cpmanifest_path)==cpmanifest
else:
    dump(cpmanifest_path,cpmanifest)
model_archive=r.parent/'matched_readout_checkpoints_20261116.tar.gz'
model_metadata=r.parent/(model_archive.name+'.json')
if model_archive.exists():
    model=read(model_metadata)
    assert sha(model_archive)==model['archive_sha256'] and model_archive.stat().st_size==model['archive_bytes']
    assert model['files']==cpmanifest
else:
    with tarfile.open(model_archive,'w:gz',compresslevel=3) as t:
        for p in checkpoint_paths:t.add(p,arcname=str(p.relative_to(r)))
    model={'archive_name':model_archive.name,'remote_archive':str(model_archive),'archive_sha256':sha(model_archive),
           'archive_bytes':model_archive.stat().st_size,'archive_files':len(checkpoint_paths),'files':cpmanifest}
    dump(model_metadata,model)
paths=[p for p in sorted(r.rglob('*')) if p.is_file() and not set(p.relative_to(r).parts)&{'scratch','__pycache__','原始运行记录'}
       and p.name!='file_checksums.json' and not p.name.endswith(('.tar.gz','.pyc'))]
manifest={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':sha(p),
          'archive_group':'checkpoints' if p in checkpoint_paths else 'evidence'} for p in paths}
dump(r/'file_checksums.json',manifest)
evidence_paths=[p for p in paths if p not in checkpoint_paths]
archive=r.parent/args.evidence_name
assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
    for p in evidence_paths+[r/'file_checksums.json']:t.add(p,arcname=str(p.relative_to(r)))
j={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_root':str(r),
   'manifest_files':len(manifest),'evidence':{'archive_name':archive.name,'remote_archive':str(archive),
   'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'archive_files':len(evidence_paths)+1},
   'checkpoints':model,'gpu_after_run':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
   'Agent_optimizer_updates':0,'readout_optimizer_updates':1280,'cache_states':2100,'new_cache_states':1890,
   'test_image_encodings':1260,'quantity_scores':340}
dump(r.parent/(archive.name+'.json'),j)
print('MATCHED_EVIDENCE_PACKAGED',len(manifest),archive.stat().st_size,model_archive.stat().st_size,flush=True)
