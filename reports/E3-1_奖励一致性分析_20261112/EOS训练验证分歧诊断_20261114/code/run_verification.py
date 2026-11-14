import json
from pathlib import Path
import subprocess
import sys
import time
r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[];error=None
try:
    for args in [['verify_diagnostic.py','--root',str(r),'--out',str(r/'服务器独立核验.json')],['audit_diagnostic_sources.py']]:
        t=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/args[0]),*args[1:]],cwd=r)
        phases.append({'args':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-t})
        if p.returncode:raise RuntimeError('Diagnostic verification failed: '+args[0])
except Exception as e:error=str(e)
(r/'post_result.json').write_text(json.dumps({'status':'complete' if error is None else 'failed','error':error,'phases':phases,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
