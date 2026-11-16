"""Durable evaluation queue waits for the active versioned generation result."""
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'evaluation_code_hashes.json').items():assert sha(r/n)==h,n
assert not (r/'evaluation_job_result.json').exists()
while not (r/'generation_job_result_v2.json').exists():time.sleep(10)
assert read(r/'generation_job_result_v2.json')['status']=='complete'
with (r/'evaluate_states.py.log').open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code/evaluate_states.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
dump(r/'evaluation_job_result.json',{'status':'complete' if p.returncode==0 else 'failed','exit_code':p.returncode})
sys.exit(p.returncode)
