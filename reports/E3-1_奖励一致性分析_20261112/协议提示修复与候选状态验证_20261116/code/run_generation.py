"""Execute sealed preparation and generation with durable status and logs."""
import os
from pathlib import Path
import subprocess
import sys
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'generation_code_hashes.json').items():assert sha(r/n)==h,n
assert not (r/'generation_job_result.json').exists()
for entry in ['prepare_inputs.py','generate_stage.py']:
 with (r/(entry+'.log')).open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code'/entry)],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
 if p.returncode:dump(r/'generation_job_result.json',{'status':'failed','entry':entry,'exit_code':p.returncode});sys.exit(p.returncode)
dump(r/'generation_job_result.json',{'status':'complete','generation':read(r/'generation_result.json')})
