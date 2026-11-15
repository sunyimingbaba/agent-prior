import datetime
import os
from pathlib import Path
import subprocess
import sys
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent;steps=[['preflight_stage.py'],['generate_policy.py','--policy-index','0'],['generate_policy.py','--policy-index','1'],['score_grid.py']];results=[];status='complete';error=None
for args in steps:
    p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],env={**os.environ,'PYTHONUTF8':'1'})
    results.append({'script':args,'exit_code':p.returncode})
    if p.returncode:status='failed';error=args;break
dump(r/'gpu_pipeline_result.json',{'status':status,'error':error,'steps':results,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
assert status=='complete',error
