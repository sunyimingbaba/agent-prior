import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[];error=None
try:
    assert sys.flags.utf8_mode==1
    deadline=time.monotonic()+7200
    while not (r/'control_result.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Control evaluation exceeded two-hour wait')
        time.sleep(10)
    assert json.loads((r/'control_result.json').read_text())['status']=='complete'
    assert json.loads((r/'tokenization_gate/result.json').read_text())['ok']
    for n,h in json.loads((r/'training_code_hashes.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
    for args in [['train_format.py'],['evaluate_supervision.py','--arm','sft_strict'],['evaluate_supervision.py','--arm','sft_legacy_greedy']]:
        tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
        phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
        if p.returncode:raise RuntimeError('Training/evaluation phase failed: '+args[0])
except Exception as e:
    error=str(e)
(r/'training_pipeline_result.json').write_text(json.dumps({'status':'complete' if error is None else 'failed','error':error,
    'phases':phases,'elapsed_seconds_including_wait':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
