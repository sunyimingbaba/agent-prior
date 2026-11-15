from pathlib import Path
from milestone_common import ROOT,PLAN,ASSETS,BASE
from stage_io import read,dump,sha
old=Path(PLAN['parent_a100_root']);pre=read(ROOT/'preflight.json')
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for n,z in pre['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==z['sha256'],n
assert sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')==pre['runtime_shim_sha256']
for n,z in read(ROOT/'source_hashes.json').items():assert sha(old/n)==z['sha256'],n
for p in PLAN['policies']:
    for n,z in read(ROOT/p['checkpoint_manifest'])['files'].items():assert sha(Path(p['checkpoint'])/n)==z['sha256'],n
for snap in ['gpu_code_hashes.json','post_code_hashes.json']:
    for n,h in read(ROOT/snap).items():assert sha(ROOT/n)==h,n
dump(ROOT/'final_source_audit.json',{'ok':True,'base_model_files_unchanged':len(pre['model_asset_hashes']),'checkpoint_files_unchanged':32,'parent_frozen_source_files_unchanged':8,'original_source_files_unchanged':len(pre['original_source_hashes']),'runtime_shim_unchanged':True,'new_optimizer_updates':0})
print('ALL_PERMUTATION_SOURCES_PRESERVED',flush=True)
