import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent
start=time.monotonic();phases=[]
for args in [['test_contract.py'],['preflight.py'],['train_corrected.py','--arm','smoke']]:
    t=time.monotonic()
    process=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
    phases.append({'args':args,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-t})
    if process.returncode:break
result={'status':'complete' if len(phases)==3 and phases[-1]['exit_code']==0 else 'failed',
        'phases':phases,'elapsed_seconds':time.monotonic()-start,
        'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'gate_result.json').write_text(json.dumps(result,indent=2)+'\n')
if result['status']!='complete':sys.exit(1)
