import argparse
import datetime
from pathlib import Path
import tarfile
from stage_io import ROOT,read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--name',default='balanced_quantity32_checkpoint_20261115.tar.gz');a=p.parse_args();assert Path(a.name).name==a.name
result=read(ROOT/'training/result.json');assert result['ok'] and result['optimizer_updates']==32
folder=ROOT/'training/final/checkpoint';manifest=read(ROOT/'training/final_checkpoint_manifest.json')
for n,i in manifest['files'].items():assert sha(folder/n)==i['sha256'],n
archive=ROOT.parent/a.name;assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
    for n in sorted(manifest['files']):t.add(folder/n,arcname=n)
out={'ok':True,'archive_name':a.name,'remote_archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files':manifest['files'],
 'lora_tensor_hash':result['final_parameter_hash'],'optimizer_updates':32,'initial_policy_lora_tensor_hash':result['initial_parameter_hash'],'includes_optimizer_and_rng_state':True,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(ROOT/'模型备份清单.json',out);print(__import__('json').dumps({k:v for k,v in out.items() if k!='files'}),flush=True)
