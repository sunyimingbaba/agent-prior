#!/usr/bin/env python3
"""Sequential inference phases; preserve exit code and phase on failure."""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
start = time.monotonic()
result = {'status': 'running', 'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phases': []}
for script in ['generate_candidates.py', 'score_candidates.py', 'analyze_pilot.py']:
    print('PHASE_START', script, flush=True)
    tick = time.monotonic()
    completed = subprocess.run([sys.executable, '-u', str(root/script)], cwd=root)
    result['phases'].append({'script': script, 'exit_code': completed.returncode, 'elapsed_seconds': time.monotonic()-tick})
    if completed.returncode:
        result.update(status='failed', failed_phase=script, elapsed_seconds=time.monotonic()-start)
        (root/'run_result.json').write_text(json.dumps(result, indent=2)+'\n')
        sys.exit(completed.returncode)
result.update(status='complete', elapsed_seconds=time.monotonic()-start, completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/'run_result.json').write_text(json.dumps(result, indent=2)+'\n')
print('PILOT_COMPLETE', json.dumps(result), flush=True)
