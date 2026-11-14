import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

from milestone_common import ROOT,BASE,ASSETS,SOURCE,PLAN,sha,dump,setup,seed
torch=setup()
mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip()
assert ' /private gpfs' in mount
for name,digest in PLAN['inputs_sha256'].items():assert sha(ROOT/name)==digest
for name,digest in PLAN['shared_code_sha256'].items():assert sha(ROOT/'code'/name)==digest
assert sha(SOURCE/'plan.json')==PLAN['source_plan_sha256']
manifest=json.loads((SOURCE/'base_input_manifest.json').read_text())
for name,info in manifest['model_asset_hashes'].items():
    path=ASSETS/'models/Qwen2.5-VL-7B-Instruct'/name
    assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256']
for path,digest in [(ASSETS/'ucm_main_lora/adapter_model.safetensors',PLAN['model_hashes']['lora_weights_sha256']),
    (ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256']),
    (ASSETS/'ucm_scene_cls.pt',PLAN['model_hashes']['scene_head_sha256'])]:assert sha(path)==digest
for split in ['train','val']:assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==manifest['arrow_sha256'][split]
assert torch.__version__=='2.4.1+cu124' and metadata.version('transformers')=='4.57.2'
site_hash=sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')
assert site_hash=='5dae4359905bc6d44c11eaaa957df8ec85f3975fe6173c35ad54164f055e6026'
seed(0)
x=torch.randn(16,16,device='cuda',requires_grad=True)
loss=(x@x.T).square().mean();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
sources={str(BASE/'CUP'/name):sha(BASE/'CUP'/name) for name in
    ['agent/rollout.py','agent/grpo/train_grpo.py','agent/grpo/scorer.py','agent/grpo/reward.py','agent/tools.py']}
dump(ROOT/'resume_preflight.json',{'ok':True,'mount':mount,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'gpu':torch.cuda.get_device_name(),'seed0_kernel_loss':float(loss),'finite_loss_and_gradient':True,
    'warm_reuse':True,'environment_modified':False,'sitecustomize_sha256':site_hash,
    'packages':{n:metadata.version(n) for n in ['torch','transformers','peft','numpy','pyarrow']},
    'original_source_hashes':sources,'model_asset_hashes':manifest['model_asset_hashes']})
print('PREFLIGHT_PASSED',flush=True)

pause_manifest=json.loads((ROOT/'pause_file_checksums.json').read_text())
for name,info in pause_manifest.items():assert sha(ROOT/name)==info['sha256'],name
assert json.loads((ROOT/'pause_status.json').read_text())['completed_full_run_records']==160
print('PAUSE_SNAPSHOT_483_FILES_UNCHANGED',flush=True)
