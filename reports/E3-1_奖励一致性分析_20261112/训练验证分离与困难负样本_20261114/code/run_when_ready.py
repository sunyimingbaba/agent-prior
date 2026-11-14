"""Wait for the dataset/software/one-update gates, then run the fixed pipeline."""
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent.parent;deadline=time.monotonic()+1800
while not (r/'gate_result.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Gates not complete within 30 minutes')
    time.sleep(5)
assert json.loads((r/'gate_result.json').read_text())['status']=='complete'
print('ALL_GATES_PASSED_STARTING_FIXED_PIPELINE',flush=True)
sys.exit(subprocess.run([sys.executable,str(r/'code/run_pipeline.py')],cwd=r).returncode)
