import importlib.metadata as md
from pathlib import Path
import shutil
import subprocess
import torch
from milestone_common import ROOT,PLAN,ASSETS,BASE,setup,seed
from stage_io import read,dump,sha
setup();old=Path(PLAN['parent_a100_root']);prior=read(ROOT/'inputs/parent_preflight.json')
mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip();assert ' /private gpfs' in mount
assert torch.cuda.get_device_name()=='NVIDIA A100-SXM4-80GB' and torch.cuda.get_device_capability()==(8,0)
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
for n,h in PLAN['inputs_sha256'].items():assert sha(ROOT/n)==h,n
for n,h in read(ROOT/'gpu_code_hashes.json').items():assert sha(ROOT/n)==h,n
for n,z in read(ROOT/'source_hashes.json').items():assert sha(old/n)==z['sha256'],n
for n,h in prior['original_source_hashes'].items():assert sha(n)==h,n
for n,z in prior['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==z['sha256'],n
for n,value in prior['packages'].items():assert md.version(n)==value,n
assert sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256'] and sha(ASSETS/'models/ViT-L-14.pt')==PLAN['model_hashes']['clip_sha256'] and sha(ASSETS/'ucm_scene_cls.pt')==PLAN['model_hashes']['scene_head_sha256']
for p in PLAN['policies']:
    for n,z in read(ROOT/p['checkpoint_manifest'])['files'].items():assert sha(Path(p['checkpoint'])/n)==z['sha256'],n
runtime=ROOT/'runtime_evidence';runtime.mkdir();shutil.copy2(BASE/'env/pyvenv.cfg',runtime/'pyvenv.cfg');shim=BASE/'env/lib/python3.12/site-packages/sitecustomize.py';assert sha(shim)==sha(old/'runtime_evidence/sitecustomize.py');shutil.copy2(shim,runtime/'sitecustomize.py')
shutil.copytree(old/'features',ROOT/'features');sc=read(old/'scoring/all_scores.json');ts=torch.load(old/'scoring/tensors.pt',map_location='cpu',weights_only=True);assert ts['record_ids']==[x['id'] for x in sc];records=[];indices=[]
for i,x in enumerate(sc):
    if x['arm'] not in ['before','after'] or x['image'] not in PLAN['images']:continue
    policy=PLAN['policies'][0 if x['arm']=='before' else 1]['name'];sf=ROOT/'historical/priors'/(x['id']+'.pt');tf=ROOT/'historical/traces'/(x['id']+'.json')
    for rel,out,h in [(x['state_file'],sf,x['state_file_sha256']),(x['trace_file'],tf,x['trace_sha256'])]:
        path=old/rel;assert sha(path)==h;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
    records.append({**x,'policy':policy,'parent_state_file':x['state_file'],'parent_trace_file':x['trace_file'],'state_file':str(sf.relative_to(ROOT)),'trace_file':str(tf.relative_to(ROOT))});indices.append(i)
assert len(records)==80 and sum(x['role']=='greedy' for x in records)==16
dump(ROOT/'historical/records.json',records);torch.save({'record_ids':[x['id'] for x in records],'image_features':ts['image_features'][indices],'quantity_similarities':ts['quantity_similarities'][indices]},ROOT/'historical/tensors.pt')
seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward();assert abs(float(loss.detach())-35.21732711791992)<1e-6 and torch.isfinite(x.grad).all()
dump(ROOT/'preflight.json',{'ok':True,'mount':mount,'gpu':torch.cuda.get_device_name(),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'packages':{n:md.version(n) for n in prior['packages']},'kernel_loss':float(loss.detach()),'kernel_gradient_finite':True,'model_asset_hashes':prior['model_asset_hashes'],'original_source_hashes':prior['original_source_hashes'],'runtime_shim_sha256':sha(shim),'checkpoint_files_verified':32,'historical_trajectories_copied':80,'historical_greedy_reused':16,'plan_sha256':sha(ROOT/'plan.json')})
print('PERMUTATION_PREFLIGHT_PASSED',flush=True)
