from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import shutil
import subprocess
import torch
from milestone_common import ROOT,PLAN,BASE,ASSETS,setup,seed
from stage_io import read,dump,sha
setup();mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip();assert ' /private gpfs' in mount
assert torch.cuda.get_device_name()=='NVIDIA A100-SXM4-80GB' and int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
old=Path(PLAN['parent_balanced_root']);parent=Path(PLAN['parent_permutation_root']);prior=read(ROOT/'inputs/parent_preflight.json')
for n,h in PLAN['inputs_sha256'].items():assert sha(ROOT/n)==h,n
for n,h in read(ROOT/'gpu_code_hashes.json').items():assert sha(ROOT/n)==h,n
for name,value in prior['packages'].items():assert md.version(name)==value,name
assert sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256'] and sha(ASSETS/'models/ViT-L-14.pt')==PLAN['model_hashes']['clip_sha256']
sourcehashes=read(parent/'runtime_evidence/utility_python_sources_after.json')['files']
for n,h in sourcehashes.items():assert sha(BASE/'CUP'/n)==h,n
snapshot=ROOT/'source_snapshot';snapshot.mkdir()
for n in ['vilt/modules/vilt_module.py','prompt_clip/model.py','agent/rollout.py']:
    p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/'CUP'/n,p)
dump(ROOT/'source_snapshot/inventory_before.json',{'files':sourcehashes,'source_root':str(BASE/'CUP'),'file_count':len(sourcehashes)})
shutil.copytree(old/'features',ROOT/'features');cases=read(ROOT/'inputs/evaluation_cases.json');by={x['image']:x for x in cases};records=[]
for c in cases:
    src=Path(PLAN['original_cache_root'])/c['historical_prior_relative_path'];assert sha(src)==c['historical_prior_sha256'];sid='original__'+c['image'];dst=ROOT/'states/priors'/(sid+'.pt');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    records.append({'id':sid,'family':'original_cached','image':c['image'],'split':c['split'],'role':'original_cached','original_first_caption':c['original_first_caption'],'state_file':str(dst.relative_to(ROOT)),'state_file_sha256':sha(dst),'source_path':str(src),'cached_policy_provenance':'[待确认：原缓存未附模型版本元数据]'})
old_scores=read(old/'scoring/all_scores.json');old_t=torch.load(old/'scoring/tensors.pt',map_location='cpu',weights_only=True);assert old_t['record_ids']==[x['id'] for x in old_scores];base_rows=[];base_ix=[]
for i,x in enumerate(old_scores):
    if x['arm'] in ['before_shared','after_shared'] or (x['arm'] in ['before','after'] and x['role']=='greedy'):base_rows.append(x);base_ix.append(i)
    if x['arm'] not in ['before','after'] or x['role']!='greedy':continue
    family='initial_SFT30' if x['arm']=='before' else 'balanced32';sid='greedy__'+family+'__'+x['image'];dst=ROOT/'states/priors'/(sid+'.pt');trace=ROOT/'states/traces'/(sid+'.json');trace.parent.mkdir(exist_ok=True)
    for src_rel,out,h in [(x['state_file'],dst,x['state_file_sha256']),(x['trace_file'],trace,x['trace_sha256'])]:assert sha(old/src_rel)==h;shutil.copy2(old/src_rel,out)
    records.append({**x,'id':sid,'parent_id':x['id'],'family':family,'role':'greedy','state_file':str(dst.relative_to(ROOT)),'trace_file':str(trace.relative_to(ROOT))})
assert len(base_rows)==340;dump(ROOT/'baseline/records.json',base_rows)
vmap={v:i for i,v in enumerate(old_t['val_record_ids'])};vrows=[x['id'] for x in base_rows if x['split']=='val'];torch.save({'record_ids':[x['id'] for x in base_rows],'image_features':old_t['image_features'][base_ix],'quantity_similarities':old_t['quantity_similarities'][base_ix],'val_record_ids':vrows,'val_similarities':old_t['val_similarities'][[vmap[v] for v in vrows]]},ROOT/'baseline/tensors.pt')
for p in PLAN['policies']:
    family=p['name'];xs=[json.loads(s) for s in (parent/'generation'/family/'records.jsonl').read_text().splitlines()];assert len(xs)==128
    for x in xs:
        sid='sample__'+x['id'];dst=ROOT/'states/priors'/(sid+'.pt');trace=ROOT/'states/traces'/(sid+'.json')
        for src_rel,out,h in [(x['state_file'],dst,x['state_file_sha256']),(x['trace_file'],trace,x['trace_sha256'])]:assert sha(parent/src_rel)==h;shutil.copy2(parent/src_rel,out)
        records.append({**x,'id':sid,'parent_id':x['id'],'family':family,'role':'candidate','state_file':str(dst.relative_to(ROOT)),'trace_file':str(trace.relative_to(ROOT))})
assert len(records)==511
for x in records:
    h=torch.load(ROOT/x['state_file'],map_location='cpu',weights_only=True);assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all();x['numeric_state_sha256']=hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()
dump(ROOT/'states/records.json',records);seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward();assert abs(float(loss.detach())-35.21732711791992)<1e-6 and torch.isfinite(x.grad).all()
dump(ROOT/'preflight.json',{'ok':True,'mount':mount,'gpu':torch.cuda.get_device_name(),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'packages':prior['packages'],'kernel_loss':float(loss.detach()),'states_copied':511,'baseline_records':340,'source_python_files_verified':len(sourcehashes),'utility_model_sha256':PLAN['utility_model_sha256'],'new_Qwen_generations':0,'optimizer_updates':0,'plan_sha256':sha(ROOT/'plan.json')});print('INJECTION_PREFLIGHT_PASSED',flush=True)
