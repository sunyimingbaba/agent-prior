from pathlib import Path
import subprocess
import sys
import os
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for n,h in read(r/'full_validation_code_hashes.json').items():assert sha(r/n)==h,n
results=[];status='complete'
for name in ['run_full_validation.py','score_full_validation.py']:
 p=subprocess.run([sys.executable,'-u',str(r/'code'/name)],env={**os.environ,'PYTHONUTF8':'1'});results.append({'script':name,'exit_code':p.returncode})
 if p.returncode:status='failed';break
dump(r/'full_validation_pipeline_result.json',{'status':status,'steps':results});assert status=='complete'
