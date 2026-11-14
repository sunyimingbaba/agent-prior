"""Recheck a warm environment without replacing the inherited preflight record."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
from milestone_common import ROOT,BASE,ASSETS,SOURCE,PLAN,sha,dump,setup,seed

assert sys.flags.utf8_mode==1 and (ROOT/'reproduction_source.json').is_file()
pre=json.loads((ROOT/'preflight.json').read_text());assert pre['ok']
mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip();assert ' /private gpfs' in mount
memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert memory<500
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for n,info in pre['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==info['sha256'],n
for n,h in PLAN['inputs_sha256'].items():assert sha(ROOT/n)==h,n
for snap in ['generation_code_hashes.json','generation_retry_code_hashes.json','generation_v3_code_hashes.json','post_code_hashes.json']:
    for n,h in json.loads((ROOT/snap).read_text()).items():assert sha(ROOT/n)==h,n
for n,version in pre['packages'].items():assert metadata.version(n)==version,n
assert sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')==pre['sitecustomize_sha256']
assert sha(Path(PLAN['policy']['checkpoint'])/'adapter_model.safetensors')==PLAN['policy']['adapter_sha256']
manifest=json.loads((SOURCE/'base_input_manifest.json').read_text())
for split in ['train','val']:assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==manifest['arrow_sha256'][split]
torch=setup();seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and abs(float(loss)-35.21732711791992)<1e-6
dump(ROOT/'reproduction_preflight.json',{'ok':True,'warm_reuse':True,'seed0_kernel_loss':float(loss),'mount':mount,
 'inherited_prefix_records':6,'environment_modified':False})
print('REPRODUCTION_WARM_ENVIRONMENT_RECHECKED')
