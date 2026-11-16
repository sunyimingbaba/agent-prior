"""Wait for frozen generation, then execute the sealed evaluation entry."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'evaluation_code_hashes.json').items():assert sha(r/n)==h,n
assert not (r/'evaluation_job_result.json').exists()
while not (r/'generation_job_result.json').exists():time.sleep(10)
assert read(r/'generation_job_result.json')['status']=='complete'
with (r/'evaluate_frozen.log').open('xb') as f:
 p=subprocess.run([sys.executable,'-u',str(r/'code/evaluate_frozen.py')],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
dump(r/'evaluation_job_result.json',{'status':'complete' if p.returncode==0 else 'failed','exit_code':p.returncode,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
sys.exit(p.returncode)
