import datetime
import json
from pathlib import Path
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS,sha,dump

pre=json.loads((ROOT/'preflight.json').read_text())
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for n,i in pre['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==i['sha256'],n
for p,h in [(ASSETS/'ucm_main_lora/adapter_model.safetensors',PLAN['model_hashes']['lora_weights_sha256']),
    (ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256']),(ASSETS/'ucm_scene_cls.pt',PLAN['model_hashes']['scene_head_sha256']),
    (ASSETS/'ucm_main_domain_tool.ckpt',PLAN['utility_model_sha256']),
    (BASE/'env/lib/python3.12/site-packages/sitecustomize.py',pre['sitecustomize_sha256']),
    (Path(PLAN['policy']['checkpoint'])/'adapter_model.safetensors',PLAN['policy']['adapter_sha256'])]:assert sha(p)==h,str(p)
preserved={}
for p in [Path(PLAN['baseline_root']),Path(PLAN['protocol_parent_root'])]:
    m=json.loads((p/'file_checksums.json').read_text())
    for n,i in m.items():assert sha(p/n)==i['sha256'],n
    preserved[p.name]=len(m)
for snapshot in ['control_code_hashes.json','training_code_hashes.json','post_code_hashes.json']:
    for n,h in json.loads((ROOT/snapshot).read_text()).items():assert sha(ROOT/n)==h,n
for label in ['intermediate_15','final']:
    m=json.loads((ROOT/'training'/(label+'_checkpoint_manifest.json')).read_text())
    for n,i in m['files'].items():assert sha(ROOT/'training'/label/'checkpoint'/n)==i['sha256'],n
backup=json.loads((ROOT/'模型备份清单.json').read_text());assert sha(backup['remote_archive'])==backup['archive_sha256']
assert json.loads((ROOT/'服务器独立核验.json').read_text())['ok']
dump(ROOT/'final_source_audit.json',{'ok':True,'original_sources_models_and_environment_unchanged':True,
 'previous_stage_files_preserved':preserved,'all_executed_code_hashes_verified':True,'new_optimizer_updates':30,
 'both_training_snapshots_preserved':True,'final_checkpoint_archive_sha256':backup['archive_sha256'],
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('ALL_ORIGINAL_SOURCES_AND_NEW_TRAINING_SNAPSHOTS_VERIFIED',flush=True)
