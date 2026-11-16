"""Sealed postprocessing queue; each command has a preserved log and exit record."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent;d=read(r/'protocol_plan.json')
for n,h in read(r/'post_code_hashes.json').items():assert sha(r/n)==h,n
assert not (r/'post_job_result.json').exists()
while not (r/'evaluation_job_result.json').exists():time.sleep(10)
assert read(r/'evaluation_job_result.json')['status']=='complete'
steps=[('analyze_states.py',[]),('verify_generation.py',['--root',str(r),'--matched',d['matched_parent'],'--out',str(r/'server_generation_verification.json')]),('verify_evaluation.py',['--root',str(r),'--parent',d['diagnostic_parent'],'--matched',d['matched_parent'],'--out',str(r/'server_evaluation_verification.json')]),('verify_descriptive.py',['--root',str(r),'--out',str(r/'server_descriptive_verification.json')]),('audit_originals.py',[])]
for entry,args in steps:
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (r/(entry+'.log')).open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code'/entry),*args],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
 row={'entry':entry,'args':args,'started_utc':started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode}
 with (r/'post_execution.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if p.returncode:dump(r/'post_job_result.json',{'status':'failed',**row});sys.exit(p.returncode)
dump(r/'post_job_result.json',{'status':'complete','generation_verification':read(r/'server_generation_verification.json'),'evaluation_verification':read(r/'server_evaluation_verification.json'),'descriptive_verification':read(r/'server_descriptive_verification.json'),'source_audit':read(r/'final_source_audit.json')})
