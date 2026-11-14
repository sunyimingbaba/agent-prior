import datetime
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
    while not (r/'generation_pipeline_result_v3.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Generation exceeded two-hour post-processing wait')
        time.sleep(10)
    assert json.loads((r/'generation_pipeline_result_v3.json').read_text())['status']=='complete'
    assert json.loads((r/'generation_complete/result.json').read_text())['ok']
    for n,h in json.loads((r/'post_code_hashes.json').read_text()).items():
        assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
    args=['score_protocol.py'];tick=time.monotonic()
    p=subprocess.run([sys.executable,str(r/'code'/args[0])],cwd=r)
    phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
    if p.returncode:raise RuntimeError('Scoring failed')
    (r/'run_result.json').write_text(json.dumps({'status':'complete','generation_entry':'generation_pipeline_result_v3.json',
        'optimization_updates':0,'selected_records':500,'all_unique_records':626},indent=2)+'\n',encoding='utf-8')
    for args in [['replay_protocol.py'],['analyze_quantity.py','--root',str(r)],['analyze_protocol.py','--root',str(r)],
                 ['verify_protocol.py','--root',str(r),'--out',str(r/'服务器独立核验.json')],['audit_protocol_sources.py']]:
        tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
        phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
        if p.returncode:raise RuntimeError('Post phase failed: '+args[0])
except Exception as e:
    error=str(e)
j={'status':'complete' if error is None else 'failed','error':error,'phases':phases,
   'elapsed_seconds_including_wait':time.monotonic()-start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'post_result.json').write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
