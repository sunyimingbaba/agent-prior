from pathlib import Path
import subprocess
import sys
import os
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
while not (r/'full_validation_pipeline_result.json').exists():time.sleep(5)
assert read(r/'full_validation_pipeline_result.json')['status']=='complete'
for n,h in read(r/'full_validation_post_hashes.json').items():assert sha(r/n)==h,n
results=[];status='complete'
for args in [['analyze_full_validation.py'],['verify_full_validation.py','--root',str(r),'--out',str(r/'full_validation/服务器独立核验.json')]]:
    p=subprocess.run([sys.executable,'-u',str(r/'code'/args[0]),*args[1:]],env={**os.environ,'PYTHONUTF8':'1'});results.append({'script':args,'exit_code':p.returncode})
    if p.returncode:status='failed';break
dump(r/'full_validation_post_result.json',{'status':status,'steps':results});assert status=='complete'
