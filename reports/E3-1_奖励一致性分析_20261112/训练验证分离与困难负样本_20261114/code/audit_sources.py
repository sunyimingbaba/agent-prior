"""Verify the original environment, inputs and executed code after the run."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

r=Path(__file__).resolve().parent.parent
assert str(r).startswith('/private/')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

plan=json.loads((r/'plan.json').read_text());base=Path(plan['base_root']);assets=base/'assets'
pre=json.loads((r/'preflight.json').read_text())
for p,h in pre['original_source_hashes'].items():assert sha(p)==h
for name,info in pre['model_asset_hashes'].items():assert sha(assets/'models/Qwen2.5-VL-7B-Instruct'/name)==info['sha256']
for p,h in [(assets/'ucm_main_lora/adapter_model.safetensors',plan['model_hashes']['lora_weights_sha256']),
            (assets/'models/ViT-L-14.pt',plan['model_hashes']['clip_sha256']),
            (assets/'ucm_scene_cls.pt',plan['model_hashes']['scene_head_sha256']),
            (assets/'ucm_main_domain_tool.ckpt',plan['utility_model_sha256']),
            (assets/'ucm/ucm_caption_karpathy_label_val.arrow',json.loads((r/'inputs/val_metadata.json').read_text())['arrow_sha256']),
            (assets/'ucm/ucm_caption_karpathy_label_train.arrow',json.loads((r/'inputs/train_metadata.json').read_text())['arrow_sha256']),
            (base/'env/lib/python3.12/site-packages/sitecustomize.py',pre['sitecustomize_sha256'])]:assert sha(p)==h
for f in ['gate_code_hashes.json','pipeline_code_hashes.json','post_code_hashes.json']:
    for name,h in json.loads((r/f).read_text()).items():assert sha(r/name)==h
assert json.loads((r/'post_result.json').read_text())['status']=='complete'
out={'ok':True,'original_sources_unchanged':pre['original_source_hashes'],
     'original_model_assets_train_val_arrows_and_environment_shim_unchanged':True,
     'pipeline_and_post_code_unchanged':True,
     'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
     'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'final_source_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
