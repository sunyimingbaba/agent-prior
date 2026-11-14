import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;deadline=time.monotonic()+1800
while not (r/'gate_result.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Gates did not finish within 30 minutes')
    time.sleep(5)
assert json.loads((r/'gate_result.json').read_text())['status']=='complete'
print('ALL_GATES_PASSED_STARTING_FROZEN_POLICY_DIAGNOSTIC',flush=True)
start=time.monotonic();commands=[['generate_quantity.py','--mode','all'],['score_quantity.py']];phases=[]
for args in commands:
    tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
    phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
    if p.returncode:break
j={'status':'complete' if len(phases)==len(commands) and phases[-1]['exit_code']==0 else 'failed',
   'phases':phases,'optimization_updates':0,'elapsed_seconds':time.monotonic()-start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'run_result.json').write_text(json.dumps(j,indent=2)+'\n')
if j['status']!='complete':sys.exit(1)
