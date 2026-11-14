import argparse
import datetime
import json
from pathlib import Path
import tarfile
from milestone_common import ROOT,PLAN,sha,dump

result=json.loads((ROOT/'training/result.json').read_text());assert result['ok'] and result['optimizer_updates']==30
folder=ROOT/'training/final/checkpoint';saved=json.loads((ROOT/'training/final_checkpoint_manifest.json').read_text())
for n,i in saved['files'].items():assert sha(folder/n)==i['sha256'],n
parser=argparse.ArgumentParser();parser.add_argument('--name',default='full_tool_supervision_checkpoint_20261114.tar.gz');args=parser.parse_args()
name=args.name;assert Path(name).name==name
archive=ROOT.parent/name;assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
    for n in sorted(saved['files']):t.add(folder/n,arcname=n)
out={'ok':True,'archive_name':name,'remote_archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
 'lora_tensor_hash':result['final_parameter_hash'],'optimizer_updates':30,'files':saved['files'],
 'includes_optimizer_and_rng_state':True,'initial_policy_lora_tensor_hash':PLAN['policy']['lora_tensor_hash'],
 'base_root_reference':PLAN['base_root'],'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(ROOT/'模型备份清单.json',out);print(json.dumps({k:v for k,v in out.items() if k!='files'}),flush=True)
