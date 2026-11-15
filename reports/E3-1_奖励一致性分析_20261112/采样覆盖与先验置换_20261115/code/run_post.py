import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
while not (r/'gpu_pipeline_result.json').exists():time.sleep(10)
assert read(r/'gpu_pipeline_result.json')['status']=='complete'
for n,h in read(r/'post_code_hashes.json').items():assert sha(r/n)==h,n
steps=[['analyze_stage.py'],['verify_stage.py','--root',str(r),'--out',str(r/'服务器独立核验.json')],['audit_sources.py']];results=[];status='complete';error=None
for args in steps:
    p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],env={**os.environ,'PYTHONUTF8':'1'})
    results.append({'script':args,'exit_code':p.returncode})
    if p.returncode:status='failed';error=args;break
dump(r/'post_result.json',{'status':status,'error':error,'steps':results,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});assert status=='complete',error
