"""Finish replay verification and analysis once the sequential GPU pipeline exits."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent
start=time.monotonic();phases=[];error=None
try:
    deadline=time.monotonic()+7200
    while not (r/'run_result.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Core pipeline result not available within two hours.')
        time.sleep(10)
    assert json.loads((r/'run_result.json').read_text())['status']=='complete'
    for args in [['verify_training_rewards.py'],['analyze_milestone.py','--root',str(r)],
                 ['verify_milestone.py','--root',str(r),'--out',str(r/'服务器独立核验.json')]]:
        tick=time.monotonic()
        process=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
        phases.append({'args':args,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-tick})
        if process.returncode:raise RuntimeError('Post phase failed: '+args[0])
except Exception as e:
    error=str(e)
result={'status':'complete' if error is None else 'failed','error':error,'phases':phases,
        'elapsed_seconds_including_wait':time.monotonic()-start,
        'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'post_result.json').write_text(json.dumps(result,indent=2)+'\n')
if error:sys.exit(1)
