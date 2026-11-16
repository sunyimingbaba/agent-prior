from pathlib import Path
import os
import subprocess
import sys
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent;assert read(r/'preflight.json')['ok'] and read(r/'gpu_pipeline_result.json')['status']=='failed'
for n,h in read(r/'gpu_retry_code_hashes.json').items():assert sha(r/n)==h,n
p=subprocess.run([sys.executable,'-u',str(r/'code/run_audit_v2.py')],env={**os.environ,'PYTHONUTF8':'1'});dump(r/'gpu_retry_result.json',{'status':'complete' if p.returncode==0 else 'failed','exit_code':p.returncode,'preflight_reused':True});assert p.returncode==0
