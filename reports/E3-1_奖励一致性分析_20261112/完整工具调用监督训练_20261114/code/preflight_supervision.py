import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS,SOURCE,sha,dump,setup,seed

torch=setup();assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip();assert ' /private gpfs' in mount
old=Path(PLAN['baseline_root']);parent=Path(PLAN['protocol_parent_root']);preserved={}
for path in [old,parent]:
    m=json.loads((path/'file_checksums.json').read_text())
    for n,i in m.items():assert sha(path/n)==i['sha256'],n
    preserved[path.name]=len(m)
assert sha(old/'plan.json')==PLAN['baseline_plan_sha256'] and sha(parent/'plan.json')==PLAN['protocol_parent_plan_sha256']
assert sha(old/'scoring/scores.json')==PLAN['baseline_scoring_sha256']
for n,h in PLAN['inputs_sha256'].items():assert sha(ROOT/n)==h,n
for n,h in PLAN['shared_code_sha256'].items():assert sha(ROOT/'code'/n)==h,n
for n,h in json.loads((ROOT/'control_code_hashes.json').read_text()).items():assert sha(ROOT/n)==h,n
manifest=json.loads((SOURCE/'base_input_manifest.json').read_text())
for n,i in manifest['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==i['sha256'],n
for p,h in [(ASSETS/'ucm_main_lora/adapter_model.safetensors',PLAN['model_hashes']['lora_weights_sha256']),
   (ASSETS/'models/ViT-L-14.pt',PLAN['model_hashes']['clip_sha256']),(ASSETS/'ucm_scene_cls.pt',PLAN['model_hashes']['scene_head_sha256']),
   (ASSETS/'ucm_main_domain_tool.ckpt',PLAN['utility_model_sha256']),
   (Path(PLAN['policy']['checkpoint'])/'adapter_model.safetensors',PLAN['policy']['adapter_sha256'])]:assert sha(p)==h,str(p)
for split in ['train','val']:assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==manifest['arrow_sha256'][split]
site=sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py');assert site=='5dae4359905bc6d44c11eaaa957df8ec85f3975fe6173c35ad54164f055e6026'
assert metadata.version('torch')=='2.4.1+cu124' and metadata.version('transformers')=='4.57.2'
seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and abs(float(loss)-35.21732711791992)<1e-6
shutil.copytree(old/'features',ROOT/'features');(ROOT/'baseline_evidence').mkdir()
archive=old.parent/'quantity_counterfactual_evidence_20261114.tar.gz';assert sha(archive)==PLAN['baseline_archive_sha256'];shutil.copy2(archive,ROOT/'baseline_evidence'/archive.name)
for src,dst in [('plan.json','plan.json'),('scoring/scores.json','scores.json'),('scoring/tensors.pt','scoring_tensors.pt'),('analysis.json','analysis.json'),('replay_verification/actual_tool_replays.json','actual_tools.json')]:shutil.copy2(old/src,ROOT/'baseline_evidence'/dst)
teachers=json.loads((ROOT/'inputs/teacher_sources.json').read_text());scores=json.loads((old/'scoring/scores.json').read_text());by={x['id']:x for x in scores}
for teacher in teachers:
    ref=by[teacher['source_record_id']];assert teacher['image']==ref['image'] and ref['quality_usable']
    assert sha(old/ref['trace_file'])==teacher['source_trace_sha256']==ref['trace_sha256']
    t=json.loads((old/ref['trace_file']).read_text());body=re.search(r'<answer>(.*?)</answer>',t['rounds'][-1]['generated_text'],re.S).group(1).strip()
    assert body==teacher['source_body']
sources={str(BASE/'CUP'/n):sha(BASE/'CUP'/n) for n in ['agent/rollout.py','agent/grpo/train_grpo.py','agent/grpo/scorer.py','agent/grpo/reward.py','agent/tools.py']}
dump(ROOT/'preflight.json',{'ok':True,'mount':mount,'warm_reuse':True,'environment_modified':False,'gpu':torch.cuda.get_device_name(),
 'seed0_kernel_loss':float(loss),'source_files_preserved':preserved,'teachers_checked':len(teachers),'original_source_hashes':sources,
 'model_asset_hashes':manifest['model_asset_hashes'],'sitecustomize_sha256':site,
 'packages':{n:metadata.version(n) for n in ['torch','transformers','peft','numpy','pyarrow']},'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('SUPERVISION_PREFLIGHT_PASSED',flush=True)
