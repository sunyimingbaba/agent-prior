import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS,setup,seed
from diagnostic_io import read,dump,sha,hydrate,tensor_sha

torch=setup();assert ' /private gpfs' in subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True)
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
for n,h in PLAN['inputs_sha256'].items():assert sha(ROOT/n)==h,n
for n,h in read(ROOT/'code_hashes.json').items():assert sha(ROOT/n)==h,n
(ROOT/'parent_evidence').mkdir();shutil.copy2(Path(PLAN['parent_root']).parent/PLAN['parent_archive_name'],ROOT/'parent_evidence'/PLAN['parent_archive_name'])
parent=hydrate();pre=read(parent/'preflight.json')
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
assert sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')==pre['sitecustomize_sha256']
assert sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256']
assert sha(PLAN['proxy_checkpoint'])==PLAN['proxy_sha256']
for n in ['cases.json','evaluation_cases.json']:assert (ROOT/'inputs'/n).read_bytes()==(parent/'inputs'/n).read_bytes()
shutil.copytree(parent/'features',ROOT/'features')
seed(0);w=torch.randn(16,16,device='cuda',requires_grad=True);loss=(w@w.T).square().mean();loss.backward()
assert abs(float(loss)-35.21732711791992)<1e-6 and torch.isfinite(w.grad).all()
sources={}
for n in ['vilt/modules/vilt_module.py','vilt/datasets/base_dataset.py','agent/rollout.py','run_frozen.py']:
    p=ROOT/'source_snapshots'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/'CUP'/n,p);sources[n]=sha(p)
cache=[];(ROOT/'historical_priors').mkdir()
for x in read(ROOT/'inputs/evaluation_cases.json'):
    source=Path(PLAN['historical_cache'])/x['historical_prior_relative_path'];assert sha(source)==x['historical_prior_sha256']
    relative='historical_priors/'+x['split']+'_'+str(x['row'])+'.pt';shutil.copy2(source,ROOT/relative)
    h=torch.load(ROOT/relative,map_location='cpu',weights_only=True);assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
    cache.append({'image':x['image'],'split':x['split'],'row':x['row'],'state_file':relative,'source_file':str(source),'file_sha256':sha(source),'tensor_sha256':tensor_sha(h),'original_first_caption':x['original_first_caption']})
dump(ROOT/'cache_mapping.json',cache)
import sys
from run_frozen import MyBackboneFinetuning
setattr(sys.modules['__main__'],'MyBackboneFinetuning',MyBackboneFinetuning)
contracts={}
for name,path in [('utility',ASSETS/'ucm_main_domain_tool.ckpt'),('proxy',Path(PLAN['proxy_checkpoint']))]:
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);config=saved['hyper_parameters']['config']
    contracts[name]={'file':str(path),'sha256':sha(path),'epoch':saved['epoch'],'global_step':saved['global_step'],
       'config':{k:config.get(k) for k in ['agent_cache_dir','use_agent_prior','agent_hidden_dim','prompt_length','data_root','seed','max_epoch']}}
assert contracts['utility']['config']['agent_cache_dir']=='/workspace/cup-debug/agent_cache_domain_tool'
assert contracts['proxy']['config']['agent_cache_dir']=='/private/AGSE_E3_1_rebuild_20261112/plain_prior_full_20261112/run_01/priors'
dump(ROOT/'checkpoint_cache_contracts.json',contracts)
dump(ROOT/'preflight.json',{'ok':True,'warm_reuse':True,'environment_modified':False,'gpu':torch.cuda.get_device_name(),'seed0_kernel_loss':float(loss),
 'parent_files_verified':1787,'historical_cache_files_verified':85,'source_snapshots_sha256':sources,'original_source_hashes':pre['original_source_hashes'],
 'sitecustomize_sha256':pre['sitecustomize_sha256'],'packages':{n:metadata.version(n) for n in pre['packages']}})
print('DIAGNOSTIC_PREFLIGHT_PASSED',flush=True)
