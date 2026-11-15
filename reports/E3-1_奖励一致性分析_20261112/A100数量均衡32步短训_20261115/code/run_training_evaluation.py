import json
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,sha
r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[];error=None
try:
    assert read(r/'prepare_result.json')['status']=='complete'
    for n,h in read(r/'training_code_hashes.json').items():assert sha(r/n)==h,n
    for script in ['train_balanced.py','evaluate_after.py']:
        t=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/script)],cwd=r);phases.append({'script':script,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-t})
        if p.returncode:raise RuntimeError(script+' failed')
except Exception as e:error=str(e)
(r/'training_pipeline_result.json').write_text(json.dumps({'status':'complete' if error is None else 'failed','error':error,'phases':phases,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
