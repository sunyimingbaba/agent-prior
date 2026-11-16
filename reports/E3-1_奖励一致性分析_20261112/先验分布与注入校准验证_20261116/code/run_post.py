from pathlib import Path
import os
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
while not (r/'gradient_pipeline_result.json').exists():time.sleep(5)
assert read(r/'gradient_pipeline_result.json')['status']=='complete' and read(r/'gpu_retry_result.json')['status']=='complete'
for n,h in read(r/'post_code_hashes.json').items():assert sha(r/n)==h,n
steps=[['verify_parent_sources.py'],['audit_sources.py'],['analyze_stage.py'],['verify_stage.py','--root',str(r),'--out',str(r/'服务器独立核验.json')]];results=[];status='complete'
for args in steps:
    p=subprocess.run([sys.executable,'-u',str(r/'code'/args[0]),*args[1:]],env={**os.environ,'PYTHONUTF8':'1'});results.append({'script':args,'exit_code':p.returncode})
    if p.returncode:status='failed';break
dump(r/'post_result.json',{'status':status,'steps':results});assert status=='complete'
