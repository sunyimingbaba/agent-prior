"""Sealed preparation plus stochastic pilot, each with a durable exit record."""
import os
from pathlib import Path
import subprocess
import sys
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'pilot_code_hashes.json').items():assert sha(r/n)==h,n
assert not (r/'initial_job_result.json').exists()
for entry in ['prepare_stage.py','run_pilot.py']:
 with (r/(entry+'.log')).open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code'/entry)],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
 if p.returncode:dump(r/'initial_job_result.json',{'status':'failed','entry':entry,'exit_code':p.returncode});sys.exit(p.returncode)
dump(r/'initial_job_result.json',{'status':'complete','pilot':read(r/'pilot/result.json')})
