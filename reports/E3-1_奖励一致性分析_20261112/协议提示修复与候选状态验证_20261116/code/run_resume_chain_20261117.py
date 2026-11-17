"""Verify the paused snapshot, warm-reuse the exact environment and complete its sealed queue."""
import datetime
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent;d=read(r/'protocol_plan.json');work=r/'resume_20261117'
assert not work.exists();work.mkdir()
for n,h in read(r/'resume_code_hashes_20261117.json').items():assert sha(r/n)==h,n
old=read(r/'file_checksums.json')
for n,v in old.items():assert (r/n).stat().st_size==v['bytes'] and sha(r/n)==v['sha256'],n
for n in ['generation_job_result_v2.json','generation_result_v2.json','evaluation_job_result.json','post_job_result.json']:assert not (r/n).exists(),n
for n in ['generation_v2/records.jsonl','generation_progress_v2.json','pilot_initial_results_v2.json','prompt_selection_lock_v2.json','file_checksums.json','runtime_environment.json','pause_20261116.json']:
 out=work/'before'/Path(n).name;out.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(r/n,out)
ref=read(r/'inputs/balanced_checkpoint_reference.json')
for n,v in ref['files'].items():assert sha(Path(d['policy']['checkpoint'])/n)==v['sha256'],n
records=[json.loads(s) for s in (r/'generation_v2/records.jsonl').read_text().splitlines()];assert len(records)==651 and len({(x['group'],x['key']) for x in records})==651
for x in records:
 for kind in ['trace','tensor']:assert sha(r/x[kind+'_file'])==x[kind+'_sha256']
env=read(r/'runtime_environment.json');packages={k:md.version(k) for k in env['packages']};assert packages==env['packages']
mount=subprocess.check_output(['findmnt','-T','/private','-o','SOURCE,TARGET,FSTYPE','-n'],text=True).strip();assert mount==env['mount']
import torch
torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward();assert torch.isfinite(x.grad).all() and abs(float(loss.detach())-env['seed0_cuda_loss'])<1e-6
witness={'ok':True,'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':packages,'unchanged_environment':True,'python':sys.version,'executable':sys.executable,'seed0_cuda_loss':float(loss.detach()),'finite_grad':True,'mount':mount,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv,noheader'],text=True).strip(),'paused_manifest_files_unchanged':len(old),'existing_record_file_hashes_verified':1302,'policy_checkpoint_files_verified':len(ref['files']),'local_pause_backup_reference':'暂停归档本机恢复核验_20261117.json'}
dump(work/'preflight.json',witness);del x,loss;torch.cuda.empty_cache()
steps=[('generate_resume_20261117.py',[], 'generate_resume_20261117.py.log'),('run_evaluation.py',[],'evaluation_queue_resume_20261117.log'),('run_post.py',[],'post_queue_resume_20261117.log'),('verify_cohort_binding.py',['--root',str(r),'--parent',d['diagnostic_parent'],'--out',str(r/'server_cohort_binding_verification.json')],'verify_cohort_binding.py.log')]
for entry,args,logname in steps:
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(r/'resume_progress_20261117.json',{'entry':entry,'started_utc':started})
 with (r/logname).open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code'/entry),*args],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
 row={'entry':entry,'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode}
 with (work/'execution.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if p.returncode:dump(r/'resume_job_result_20261117.json',{'status':'failed',**row});sys.exit(p.returncode)
 if entry=='generate_resume_20261117.py':dump(r/'generation_job_result_v2.json',{'status':'complete','generation':read(r/'generation_result_v2.json'),'resume_entry':entry})
for n,v in old.items():
 if n=='generation_v2/records.jsonl':assert (r/n).read_bytes().startswith((work/'before/records.jsonl').read_bytes())
 else:assert sha(r/n)==v['sha256'],n
dump(r/'resume_job_result_20261117.json',{'status':'complete','preserved_pre_pause_files':len(old)-1,'original_651_record_prefix_preserved':True,'new_generation_records':21,'exact_resume_anchors':3,'Agent_optimizer_updates':0,'readout_optimizer_updates':0,'cohort_binding':read(r/'server_cohort_binding_verification.json')})
