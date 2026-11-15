import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

r=Path(__file__).resolve().parent;spec=json.loads((r/'environment_spec.json').read_text());assert str(r)==spec['runtime_root']
start=time.monotonic();error=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=Path(spec['preserved_legacy_venv']);saved={str(p):sha(p) for p in [old/'pyvenv.cfg',old/'lib/python3.12/site-packages/sitecustomize.py']}
(r/'before_environment_hashes.json').write_text(json.dumps(saved,indent=2)+'\n')
try:
    assert not (r/'env').exists()
    subprocess.run([spec['base_python'],'-m','venv','--system-site-packages',str(r/'env')],check=True)
    site=r/'env/lib/python3.12/site-packages'
    shutil.copytree(old/'lib/python3.12/site-packages',site,dirs_exist_ok=True,symlinks=True)
    (r/'scratch').mkdir();(r/'pip_cache').mkdir()
    constraints=r/'constraints.txt';constraints.write_text('\n'.join(k+'=='+v for k,v in spec['target_versions'].items())+'\n')
    env=os.environ.copy();env.update(PIP_CACHE_DIR=str(r/'pip_cache'),TMPDIR=str(r/'scratch'),PYTHONUTF8='1')
    subprocess.run([str(r/'env/bin/python'),'-m','pip','install','torch==2.7.1','torchvision==0.22.1','torchaudio==2.7.1','--index-url',spec['torch_wheel_index'],'--constraint',str(constraints)],env=env,check=True)
    for p,h in saved.items():assert sha(p)==h,p
except Exception as e:error=repr(e)
(r/'setup_result.json').write_text(json.dumps({'status':'complete' if error is None else 'failed','error':error,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
if error:raise SystemExit(error)
