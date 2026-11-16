from pathlib import Path
import os
import subprocess
import sys
from stage_io import dump
r=Path(__file__).resolve().parent.parent;results=[];status='complete'
for name in ['preflight_stage.py','run_audit.py']:
    p=subprocess.run([sys.executable,'-u',str(r/'code'/name)],env={**os.environ,'PYTHONUTF8':'1'});results.append({'script':name,'exit_code':p.returncode})
    if p.returncode:status='failed';break
dump(r/'gpu_pipeline_result.json',{'status':status,'steps':results});assert status=='complete'
