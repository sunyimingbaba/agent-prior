import datetime
import json
from pathlib import Path
import subprocess
from milestone_common import ROOT,PLAN,ASSETS,BASE,sha,dump

pre=json.loads((ROOT/'preflight.json').read_text());old=Path(PLAN['baseline_root'])
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for name,info in pre['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/name)==info['sha256'],name
for path,h in [(ASSETS/'ucm_main_lora/adapter_model.safetensors',PLAN['model_hashes']['lora_weights_sha256']),
    (ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256']),
    (ASSETS/'ucm_scene_cls.pt',PLAN['model_hashes']['scene_head_sha256']),
    (ASSETS/'ucm_main_domain_tool.ckpt',PLAN['utility_model_sha256']),
    (BASE/'env/lib/python3.12/site-packages/sitecustomize.py',pre['sitecustomize_sha256']),
    (Path(PLAN['policy']['checkpoint'])/'adapter_model.safetensors',PLAN['policy']['adapter_sha256']),
    (Path(PLAN['policy']['source_remote_archive']),PLAN['policy']['source_archive_sha256'])]:assert sha(path)==h,str(path)
previous=json.loads((old/'file_checksums.json').read_text())
for name,info in previous.items():assert sha(old/name)==info['sha256'],name
for snap in ['generation_code_hashes.json','generation_retry_code_hashes.json','generation_v3_code_hashes.json','post_code_hashes.json']:
    for n,h in json.loads((ROOT/snap).read_text()).items():assert sha(ROOT/n)==h,n
assert json.loads((ROOT/'服务器独立核验.json').read_text())['ok']
dump(ROOT/'final_source_audit.json',{'ok':True,'original_sources_models_and_environment_unchanged':True,
 'previous_stage_files_preserved':len(previous),'all_executed_code_hashes_verified':True,'optimization_updates':0,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('ALL_SOURCES_AND_PREVIOUS_STAGE_PRESERVED',flush=True)
