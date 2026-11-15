import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,sha
r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[];error=None
try:
    deadline=time.monotonic()+10800
    while not (r/'training_pipeline_result.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Training/evaluation wait exceeded three hours')
        time.sleep(10)
    assert read(r/'training_pipeline_result.json')['status']=='complete'
    for n,h in read(r/'post_code_hashes.json').items():assert sha(r/n)==h,n
    for args in [['score_stage.py'],['analyze_stage.py','--root',str(r)],['verify_stage.py','--root',str(r),'--out',str(r/'服务器独立核验.json')],['audit_sources.py']]:
        t=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r);phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-t})
        if p.returncode:raise RuntimeError(args[0]+' failed')
except Exception as e:error=str(e)
(r/'post_result.json').write_text(json.dumps({'status':'complete' if error is None else 'failed','error':error,'phases':phases,'elapsed_seconds_including_wait':time.monotonic()-start,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
