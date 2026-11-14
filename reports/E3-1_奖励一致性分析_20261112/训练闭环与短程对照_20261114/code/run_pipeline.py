import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent
assert json.loads((r/'gate_result.json').read_text())['status']=='complete'
start=time.monotonic();phases=[]
commands=[['evaluate_cache.py','--policy','baseline'],['train_corrected.py','--arm','retrieval'],
          ['evaluate_cache.py','--policy','retrieval'],['train_corrected.py','--arm','format_only'],
          ['evaluate_cache.py','--policy','format_only'],['score_evaluation.py']]
for args in commands:
    tick=time.monotonic()
    process=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
    phases.append({'args':args,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-tick})
    if process.returncode:break
result={'status':'complete' if len(phases)==len(commands) and phases[-1]['exit_code']==0 else 'failed',
        'phases':phases,'elapsed_seconds':time.monotonic()-start,
        'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'run_result.json').write_text(json.dumps(result,indent=2)+'\n')
if result['status']!='complete':sys.exit(1)
