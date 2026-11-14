import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;assert not (r/'run_result.json').exists()
start=time.monotonic();commands=[['preflight_resume.py'],['resume_quantity.py'],['score_resumed_quantity.py']];phases=[]
for args in commands:
    tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
    phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
    if p.returncode:break
j={'status':'complete' if len(phases)==len(commands) and phases[-1]['exit_code']==0 else 'failed',
   'phases':phases,'optimization_updates':0,'resumed_from_records':160,'new_generation_records':340,
   'elapsed_seconds':time.monotonic()-start,'elapsed_scope':'resume pipeline only; excludes original 160 trajectories and pause',
   'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'run_result.json').write_text(json.dumps(j,indent=2)+'\n')
if j['status']!='complete':sys.exit(1)
