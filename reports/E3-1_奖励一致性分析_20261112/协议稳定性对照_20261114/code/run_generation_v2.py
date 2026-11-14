"""UTF-8 launch and explicit failure reporting; original failed entry is preserved."""
import json
import locale
import os
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;start=time.monotonic();phases=[];error=None
try:
    assert sys.flags.utf8_mode==1 and locale.getpreferredencoding(False).lower().replace('-','')=='utf8'
    assert json.loads((r/'preflight.json').read_text(encoding='utf-8'))['ok']
    assert not (r/'generation_pipeline_result_v2.json').exists()
    for script in ['test_protocol.py','generate_protocol.py']:
        tick=time.monotonic();p=subprocess.run([sys.executable,str(r/'code'/script)],cwd=r)
        phases.append({'script':script,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick})
        if p.returncode:raise RuntimeError('Generation phase failed: '+script)
except Exception as e:
    error=str(e)
(r/'generation_pipeline_result_v2.json').write_text(json.dumps({'status':'complete' if error is None else 'failed',
    'error':error,'phases':phases,'elapsed_seconds':time.monotonic()-start,'python_utf8_mode':sys.flags.utf8_mode,
    'original_failed_attempt':'failed_attempt_01','authoritative_generation_result':True},indent=2)+'\n',encoding='utf-8')
if error:sys.exit(1)
