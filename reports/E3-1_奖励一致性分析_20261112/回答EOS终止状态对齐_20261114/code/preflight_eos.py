import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS,SOURCE,sha,dump,setup,seed

torch=setup();source=Path(PLAN['parent_root']);pre=json.loads((source/'preflight.json').read_text())
mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip();assert ' /private gpfs' in mount
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
manifest=json.loads((source/'file_checksums.json').read_text());assert len(manifest)==2000
for n,i in manifest.items():assert sha(source/n)==i['sha256'],n
assert json.loads((source/'post_result.json').read_text())['status']=='complete'
assert sha(source/'plan.json')==PLAN['parent_plan_sha256']
archive=source.parent/PLAN['parent_archive_name'];assert sha(archive)==PLAN['parent_archive_sha256']
for n,h in PLAN['inputs_sha256'].items():assert sha(ROOT/n)==h,n
for n,h in PLAN['shared_code_sha256'].items():assert sha(ROOT/'code'/n)==h,n
for n,h in json.loads((ROOT/'generation_code_hashes.json').read_text()).items():assert sha(ROOT/n)==h,n
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for n,i in pre['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==i['sha256'],n
for n,i in json.loads((ROOT/'inputs/inherited_sft_model_reference.json').read_text())['files'].items():assert sha(Path(PLAN['policy']['checkpoint'])/n)==i['sha256'],n
for path,h in [(ASSETS/'ucm_main_lora/adapter_model.safetensors',PLAN['model_hashes']['lora_weights_sha256']),(ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256']),(ASSETS/'ucm_scene_cls.pt',PLAN['model_hashes']['scene_head_sha256']),(ASSETS/'ucm_main_domain_tool.ckpt',PLAN['utility_model_sha256'])]:assert sha(path)==h
base_manifest=json.loads((SOURCE/'base_input_manifest.json').read_text())
for split in ['train','val']:assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==base_manifest['arrow_sha256'][split]
assert sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')==pre['sitecustomize_sha256']
assert metadata.version('torch')=='2.4.1+cu124' and metadata.version('transformers')=='4.57.2'
seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and abs(float(loss)-35.21732711791992)<1e-6
(ROOT/'parent_evidence').mkdir();shutil.copy2(archive,ROOT/'parent_evidence'/archive.name)
shutil.copytree(source/'features',ROOT/'features')
reference=ROOT/'reference_run';reference.mkdir()
names=['scoring/all_scores.json','scoring/tensors.pt','analysis.json','inputs/evaluation_cases.json','inputs/cases.json',
 'baseline_evidence/scores.json','training/result.json','evaluation/sft_strict/settings.json','evaluation/sft_strict/result.json']
scores=json.loads((source/'scoring/all_scores.json').read_text());matched=[x for x in scores if x['arm']=='sft_strict'];assert len(matched)==425
names += [x[k] for x in matched for k in ['state_file','trace_file']]
for n in names:
    p=reference/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/n,p);assert sha(p)==manifest[n]['sha256']
dump(ROOT/'reference_file_checksums.json',{n:manifest[n] for n in names})
dump(ROOT/'preflight.json',{'ok':True,'warm_reuse':True,'environment_modified':False,'mount':mount,'gpu':torch.cuda.get_device_name(),
 'source_files_preserved':len(manifest),'reference_files_copied':len(names),'seed0_kernel_loss':float(loss),
 'original_source_hashes':pre['original_source_hashes'],'model_asset_hashes':pre['model_asset_hashes'],'sitecustomize_sha256':pre['sitecustomize_sha256'],
 'packages':{n:metadata.version(n) for n in ['torch','transformers','peft','numpy','pyarrow']},'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('EOS_ALIGNMENT_PREFLIGHT_PASSED',flush=True)
