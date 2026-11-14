import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;assert json.loads((r/'gate_result.json').read_text())['status']=='complete'
b=json.loads((r/'pilot/branch_decision.json').read_text())
commands=[['train_separated.py','--arm',arm] for arm in b['arms']]+[['evaluate_separated.py','--policy',p] for p in b['policies']]+[['score_validation.py']]
start=time.monotonic();phases=[]
for args in commands:
    tick=time.monotonic();process=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
    phases.append({'args':args,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-tick})
    if process.returncode:break
j={'status':'complete' if len(phases)==len(commands) and phases[-1]['exit_code']==0 else 'failed',
   'phases':phases,'elapsed_seconds':time.monotonic()-start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'run_result.json').write_text(json.dumps(j,indent=2)+'\n')
if j['status']!='complete':sys.exit(1)
