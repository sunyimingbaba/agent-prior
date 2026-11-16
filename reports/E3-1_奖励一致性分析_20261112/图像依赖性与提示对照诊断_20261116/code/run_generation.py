"""Durable isolated generation job; old output files are retained on failure."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'generation_code_hashes.json').items():assert sha(r/n)==h,n
assert not (r/'generation_job_result.json').exists()
for script,log in [('prepare_inputs.py','prepare_inputs.log'),('generate_interventions.py','generate_interventions.log')]:
 if script=='prepare_inputs.py' and (r/'inputs/prepared.json').exists():continue
 with (r/log).open('xb') as f:
  p=subprocess.run([sys.executable,'-u',str(r/'code'/script)],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
 if p.returncode:
  dump(r/'generation_job_result.json',{'status':'failed','entry':script,'exit_code':p.returncode,'log':log});sys.exit(p.returncode)
dump(r/'generation_job_result.json',{'status':'complete','generation':read(r/'generation_result.json'),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
