import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent
assert json.loads((r/'gate_result.json').read_text())['status']=='complete'
commands=[['train_aligned.py','--arm','aligned_margin']]+[['evaluate_alignment.py','--policy',p] for p in
    ['baseline','retrieval','format_only','aligned_margin']]+[['score_alignment.py']]
start=time.monotonic();phases=[]
for args in commands:
    t=time.monotonic();process=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
    phases.append({'args':args,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-t})
    if process.returncode:break
j={'status':'complete' if len(phases)==len(commands) and phases[-1]['exit_code']==0 else 'failed',
   'phases':phases,'elapsed_seconds':time.monotonic()-start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'run_result.json').write_text(json.dumps(j,indent=2)+'\n')
if j['status']!='complete':sys.exit(1)
