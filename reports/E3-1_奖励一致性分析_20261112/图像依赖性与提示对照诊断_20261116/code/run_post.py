"""Sealed CPU postprocessing, verification and original-source audit after frozen evaluation."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent;d=read(r/'diagnostic_plan.json')
for n,h in read(r/'post_code_hashes.json').items():assert sha(r/n)==h,n
assert not (r/'post_job_result.json').exists()
while not (r/'evaluation_job_result.json').exists():time.sleep(10)
assert read(r/'evaluation_job_result.json')['status']=='complete'
steps=[('analyze_diagnostics.py',[]),('verify_diagnostics.py',['--root',str(r),'--parent',d['matched_parent'],'--out',str(r/'server_verification.json')]),('verify_path_states.py',['--root',str(r),'--out',str(r/'server_path_state_verification.json')]),('audit_originals.py',[])]
for entry,args in steps:
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (r/(entry+'.log')).open('xb') as f:
  p=subprocess.run([sys.executable,'-u',str(r/'code'/entry),*args],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
 item={'entry':entry,'args':args,'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode}
 with (r/'post_execution.jsonl').open('a') as f:f.write(__import__('json').dumps(item)+'\n')
 if p.returncode:
  dump(r/'post_job_result.json',{'status':'failed',**item});sys.exit(p.returncode)
dump(r/'post_job_result.json',{'status':'complete','server_independent_verification':read(r/'server_verification.json'),'path_state_verification':read(r/'server_path_state_verification.json'),'final_source_audit':read(r/'final_source_audit.json')})
