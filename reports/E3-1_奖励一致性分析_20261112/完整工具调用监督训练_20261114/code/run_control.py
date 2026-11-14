import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[];error=None
try:
    assert sys.flags.utf8_mode==1
    for args in [['test_supervised_format.py'],['preflight_supervision.py'],['evaluate_supervision.py','--arm','c742_strict']]:
        tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
        phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
        if p.returncode:raise RuntimeError('Control phase failed: '+args[0])
except Exception as e:
    error=str(e)
(r/'control_result.json').write_text(json.dumps({'status':'complete' if error is None else 'failed','error':error,'phases':phases,
    'elapsed_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
