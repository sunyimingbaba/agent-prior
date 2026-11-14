import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
from milestone_common import ROOT,BASE,ASSETS,SOURCE,PLAN,sha,dump,setup,seed

torch=setup();old=Path(PLAN['baseline_root'])
mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip()
assert ' /private gpfs' in mount
assert json.loads((old/'post_result.json').read_text())['status']=='complete'
assert sha(old/'plan.json')==PLAN['baseline_plan_sha256']
assert sha(old/'scoring/scores.json')==PLAN['baseline_scoring_sha256']
previous=json.loads((old/'file_checksums.json').read_text())
for name,info in previous.items():assert sha(old/name)==info['sha256'],name
archive=old.parent/'quantity_counterfactual_evidence_20261114.tar.gz'
assert sha(archive)==PLAN['baseline_archive_sha256']
for name,h in PLAN['inputs_sha256'].items():assert sha(ROOT/name)==h,name
for name,h in PLAN['shared_code_sha256'].items():assert sha(ROOT/'code'/name)==h,name
for name,h in json.loads((ROOT/'generation_code_hashes.json').read_text()).items():assert sha(ROOT/name)==h,name
assert sha(SOURCE/'plan.json')==PLAN['source_plan_sha256']
manifest=json.loads((SOURCE/'base_input_manifest.json').read_text())
for name,info in manifest['model_asset_hashes'].items():
    path=ASSETS/'models/Qwen2.5-VL-7B-Instruct'/name
    assert path.stat().st_size==info['bytes'] and sha(path)==info['sha256']
for path,digest in [(ASSETS/'ucm_main_lora/adapter_model.safetensors',PLAN['model_hashes']['lora_weights_sha256']),
    (ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256']),
    (ASSETS/'ucm_scene_cls.pt',PLAN['model_hashes']['scene_head_sha256']),
    (ASSETS/'ucm_main_domain_tool.ckpt',PLAN['utility_model_sha256']),
    (Path(PLAN['policy']['checkpoint'])/'adapter_model.safetensors',PLAN['policy']['adapter_sha256'])]:assert sha(path)==digest
for split in ['train','val']:assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==manifest['arrow_sha256'][split]
assert metadata.version('torch')=='2.4.1+cu124' and metadata.version('transformers')=='4.57.2'
site_hash=sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')
assert site_hash=='5dae4359905bc6d44c11eaaa957df8ec85f3975fe6173c35ad54164f055e6026'
seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and abs(float(loss)-35.21732711791992)<1e-6
for folder in ['features']:
    shutil.copytree(old/folder,ROOT/folder)
(ROOT/'baseline_evidence').mkdir()
shutil.copy2(archive,ROOT/'baseline_evidence'/archive.name)
for src,dst in [('plan.json','plan.json'),('scoring/scores.json','scores.json'),('scoring/tensors.pt','scoring_tensors.pt'),
                ('analysis.json','analysis.json'),('服务器独立核验.json','server_verification.json'),('file_checksums.json','file_checksums.json')]:
    shutil.copy2(old/src,ROOT/'baseline_evidence'/dst)
sources={str(BASE/'CUP'/name):sha(BASE/'CUP'/name) for name in
 ['agent/rollout.py','agent/grpo/train_grpo.py','agent/grpo/scorer.py','agent/grpo/reward.py','agent/tools.py']}
dump(ROOT/'preflight.json',{'ok':True,'mount':mount,'warm_reuse':True,'environment_modified':False,
 'gpu':torch.cuda.get_device_name(),'seed0_kernel_loss':float(loss),'finite_loss_and_gradient':True,
 'previous_1219_files_preserved':len(previous),'original_source_hashes':sources,'sitecustomize_sha256':site_hash,
 'packages':{n:metadata.version(n) for n in ['torch','transformers','peft','numpy','pyarrow']},
 'model_asset_hashes':manifest['model_asset_hashes'],'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('PREFLIGHT_PROTOCOL_PASSED',flush=True)
