import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

r = Path(__file__).resolve().parent
start = time.monotonic()
result = {'status': 'running', 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phases': []}
for phase in ['generate_states.py', 'score_states.py', 'analyze_states.py']:
    tick = time.monotonic()
    process = subprocess.run([sys.executable, str(r/phase)], cwd=r)
    result['phases'].append({'script': phase, 'exit_code': process.returncode, 'elapsed_seconds': time.monotonic()-tick})
    if process.returncode:
        result['status'] = 'failed'
        break
else:
    result['status'] = 'complete'
result['elapsed_seconds'] = time.monotonic()-start
result['completed_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(r/'run_result.json').write_text(json.dumps(result, indent=2)+'\n')
if result['status'] != 'complete':
    sys.exit(1)
