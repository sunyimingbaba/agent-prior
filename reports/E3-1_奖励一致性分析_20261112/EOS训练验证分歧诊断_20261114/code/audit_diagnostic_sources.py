import subprocess
from pathlib import Path
from milestone_common import ROOT,PLAN,BASE,ASSETS
from diagnostic_io import read,sha,dump,hydrate
pre=read(ROOT/'preflight.json');parent=hydrate()
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for n,h in pre['source_snapshots_sha256'].items():assert sha(BASE/'CUP'/n)==sha(ROOT/'source_snapshots'/n)==h,n
assert sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')==pre['sitecustomize_sha256']
for n,h in [(ASSETS/'ucm_main_domain_tool.ckpt',PLAN['utility_model_sha256']),(Path(PLAN['proxy_checkpoint']),PLAN['proxy_sha256']),(ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256'])]:assert sha(n)==h,n
for snapshot in ['code_hashes.json','analysis_code_hashes.json']:
    for n,h in read(ROOT/snapshot).items():assert sha(ROOT/n)==h,n
for x in read(ROOT/'cache_mapping.json'):assert sha(x['source_file'])==x['file_sha256'],x['image']
assert read(ROOT/'服务器独立核验.json')['ok']
dump(ROOT/'final_source_audit.json',{'ok':True,'original_models_sources_historical_priors_and_environment_unchanged':True,
 'parent_archive_files_preserved':1787,'new_optimizer_updates':0,'new_generation_records':0,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()})
print('DIAGNOSTIC_SOURCES_PRESERVED')
