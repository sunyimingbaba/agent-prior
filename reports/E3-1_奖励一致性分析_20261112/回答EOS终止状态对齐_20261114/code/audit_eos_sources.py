import datetime
import json
from pathlib import Path
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS,SOURCE,sha,dump

pre=json.loads((ROOT/'preflight.json').read_text(encoding='utf-8'))
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for n,i in pre['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==i['sha256'],n
for p,h in [(ASSETS/'ucm_main_lora/adapter_model.safetensors',PLAN['model_hashes']['lora_weights_sha256']),
    (ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256']),(ASSETS/'ucm_scene_cls.pt',PLAN['model_hashes']['scene_head_sha256']),
    (ASSETS/'ucm_main_domain_tool.ckpt',PLAN['utility_model_sha256']),
    (BASE/'env/lib/python3.12/site-packages/sitecustomize.py',pre['sitecustomize_sha256'])]:assert sha(p)==h,str(p)
for n,i in json.loads((ROOT/'inputs/inherited_sft_model_reference.json').read_text(encoding='utf-8'))['files'].items():
    assert sha(Path(PLAN['policy']['checkpoint'])/n)==i['sha256'],n
assert sha(PLAN['policy']['source_remote_archive'])==PLAN['policy']['source_archive_sha256']
parent=Path(PLAN['parent_root']);manifest=json.loads((parent/'file_checksums.json').read_text(encoding='utf-8'))
for n,i in manifest.items():assert sha(parent/n)==i['sha256'],n
assert sha(parent/'plan.json')==PLAN['parent_plan_sha256']
assert sha(ROOT/'parent_evidence'/PLAN['parent_archive_name'])==PLAN['parent_archive_sha256']
for snapshot in ['generation_code_hashes.json','post_code_hashes.json']:
    for n,h in json.loads((ROOT/snapshot).read_text(encoding='utf-8')).items():assert sha(ROOT/n)==h,n
bm=json.loads((SOURCE/'base_input_manifest.json').read_text(encoding='utf-8'))
for split in ['train','val']:assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==bm['arrow_sha256'][split]
assert json.loads((ROOT/'服务器独立核验.json').read_text(encoding='utf-8'))['ok']
generation=json.loads((ROOT/'generation/result.json').read_text(encoding='utf-8'))
assert generation['optimization_updates']==0 and generation['lora_tensor_hash_unchanged']==PLAN['policy']['lora_tensor_hash']
dump(ROOT/'final_source_audit.json',{'ok':True,'original_sources_models_data_and_environment_unchanged':True,
 'parent_stage_files_preserved':len(manifest),'all_executed_code_hashes_verified':True,'inherited_checkpoint_files_preserved':16,
 'new_optimizer_updates':0,'policy_lora_tensor_hash':PLAN['policy']['lora_tensor_hash'],
 'inherited_model_archive_sha256':PLAN['policy']['source_archive_sha256'],
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('ALL_EOS_SOURCES_AND_INHERITED_MODEL_PRESERVED',flush=True)
