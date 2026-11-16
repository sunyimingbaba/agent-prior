"""Execute sealed preparation and generation with durable status and logs."""
import os
from pathlib import Path
import subprocess
import sys
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'generation_code_hashes_v2.json').items():assert sha(r/n)==h,n
assert not (r/'generation_job_result_v2.json').exists()
assert read(r/'inputs/prepared.json')['ok']
d=read(r/'protocol_plan.json');ref=read(r/'inputs/balanced_checkpoint_reference.json')
for n,v in ref['files'].items():assert sha(Path(d['policy']['checkpoint'])/n)==v['sha256'],n
for entry in ['generate_stage_v2.py']:
 with (r/(entry+'.log')).open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code'/entry)],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
 if p.returncode:dump(r/'generation_job_result_v2.json',{'status':'failed','entry':entry,'exit_code':p.returncode});sys.exit(p.returncode)
dump(r/'generation_job_result_v2.json',{'status':'complete','generation':read(r/'generation_result_v2.json')})
