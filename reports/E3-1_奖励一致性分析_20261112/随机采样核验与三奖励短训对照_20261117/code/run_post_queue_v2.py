"""After fixed-policy evaluation: frozen analysis and independent CPU reconstructions."""
from pathlib import Path
import os
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'post_code_hashes_v2.json').items():assert sha(r/n)==h,n
assert not (r/'post_job_result.json').exists()
while not (r/'continuation_job_result_v2.json').exists():time.sleep(10)
x=read(r/'continuation_job_result_v2.json')
if x['status']!='generation_training_evaluation_complete':dump(r/'post_job_result.json',{'status':'prior_stage_stopped','source_status':x});raise SystemExit(0)
jobs=[('analyze_comparison_v3.py',[]),('verify_raw_cpu_v2.py',[str(r)])]+[('verify_normalization_cuda.py',[str(r),arm]) for arm in read(r/'plan.json')['reward_arms'] if not (r/'training_v2'/arm/'independent_cuda_normalization.json').exists()]+[('verify_training_cpu_v2.py',[str(r)]),('verify_metrics_cpu.py',[str(r)])]
for name,args in jobs:
 with (r/('post_v2_'+name+'_'+('_'.join(args[1:]) if len(args)>1 else '')+'.log')).open('xb') as f:q=subprocess.run([sys.executable,'-u',str(r/'code'/name),*args],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
 if q.returncode:dump(r/'post_job_result.json',{'status':'failed','entry':name,'exit_code':q.returncode});raise SystemExit(q.returncode)
dump(r/'post_job_result.json',{'status':'complete','analysis_sha256':sha(r/'analysis.json'),'independent_raw_sha256':sha(r/'independent_raw_verification.json'),'independent_training_sha256':sha(r/'independent_training_verification.json'),'independent_metrics_sha256':sha(r/'independent_metric_verification.json'),'normalization_reconstruction':{arm:sha(r/'training_v2'/arm/'independent_cuda_normalization.json') for arm in read(r/'plan.json')['reward_arms']},'delivery_and_fresh_audit':'pending'})
