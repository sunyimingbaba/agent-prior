import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[];error=None
try:
    assert sys.flags.utf8_mode==1
    for script in ['test_eos_boundary.py','preflight_eos.py','generate_eos.py']:
        tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/script)],cwd=r)
        phases.append({'script':script,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
        if p.returncode:raise RuntimeError('EOS generation phase failed: '+script)
except Exception as e:
    error=str(e)
(r/'generation_pipeline_result.json').write_text(json.dumps({'status':'complete' if error is None else 'failed','error':error,
 'phases':phases,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
