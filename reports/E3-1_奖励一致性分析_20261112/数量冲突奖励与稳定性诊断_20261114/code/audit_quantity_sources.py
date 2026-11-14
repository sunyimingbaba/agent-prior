"""Audit original assets, the referenced model, and pause preservation."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

r=Path(__file__).resolve().parent.parent;assert str(r).startswith('/private/')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

plan=json.loads((r/'plan.json').read_text());base=Path(plan['base_root']);assets=base/'assets'
pre=json.loads((r/'resume_preflight.json').read_text());source=Path(plan['source_run_root'])
original=json.loads((source/'base_input_manifest.json').read_text())
for file,h in pre['original_source_hashes'].items():assert sha(file)==h
for name,info in pre['model_asset_hashes'].items():assert sha(assets/'models/Qwen2.5-VL-7B-Instruct'/name)==info['sha256']
for file,h in [(assets/'ucm_main_lora/adapter_model.safetensors',plan['model_hashes']['lora_weights_sha256']),
               (assets/'models/ViT-L-14.pt',plan['model_hashes']['clip_sha256']),
               (assets/'ucm_scene_cls.pt',plan['model_hashes']['scene_head_sha256']),
               (assets/'ucm_main_domain_tool.ckpt',plan['utility_model_sha256']),
               (base/'env/lib/python3.12/site-packages/sitecustomize.py',pre['sitecustomize_sha256'])]:assert sha(file)==h
for split in ['train','val']:assert sha(assets/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==original['arrow_sha256'][split]
for file in ['gate_code_hashes.json','pipeline_code_hashes.json','post_code_hashes.json','resume_code_hashes.json','resume_post_code_hashes.json']:
    for name,h in json.loads((r/file).read_text()).items():assert sha(r/name)==h
pause=json.loads((r/'pause_file_checksums.json').read_text())
for name,info in pause.items():assert sha(r/name)==info['sha256']
ref=plan['policy'];model_manifest=json.loads((Path(ref['checkpoint']).parents[2]/'模型备份清单.json').read_text())['policies']['format_only']
assert model_manifest['archive_sha256']==ref['source_archive_sha256']
assert sha(ref['source_remote_archive'])==ref['source_archive_sha256']
for name,info in model_manifest['files'].items():assert sha(Path(ref['checkpoint'])/name)==info['sha256']
assert json.loads((r/'post_result.json').read_text())['status']=='complete'
out={'ok':True,'original_sources_unchanged':pre['original_source_hashes'],
     'original_models_train_val_arrows_and_environment_unchanged':True,'all_executed_code_hashes_unchanged':True,
     'all_paused_files_preserved':len(pause),'source_policy_checkpoint_files_verified':len(model_manifest['files']),
     'source_policy_archive_sha256':ref['source_archive_sha256'],
     'optimization_updates':0,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
     'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'final_source_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out))
