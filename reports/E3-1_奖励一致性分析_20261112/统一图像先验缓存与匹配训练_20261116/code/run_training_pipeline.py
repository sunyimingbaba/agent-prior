"""Resume-safe sequential GPU queue; evaluate only after every fixed training arm."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read, dump, sha

r = Path(__file__).resolve().parent.parent
steps = []
def status(phase, **extra):
    dump(r/'matched_pipeline_progress.json', {'phase': phase, 'steps': steps,
         'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra})

for manifest in ['training_code_hashes_v4.json', 'evaluation_code_hashes.json', 'queue_code_hashes.json']:
    for n, h in read(r/manifest).items():
        assert sha(r/n) == h, n
assert read(r/'training_preflight.json')['ok']
status('waiting_for_cache')
while not (r/'cache_pipeline_result.json').exists():
    progress = r/'cache_progress.json'
    assert not progress.exists() or time.time()-progress.stat().st_mtime < 900, 'Cache progress stale'
    time.sleep(10)
assert read(r/'cache_pipeline_result.json')['status'] == 'complete'
assert read(r/'cache_result.json')['states'] == 2100
with (r/'cache_audit.log').open('ab') as log:
    audit = subprocess.run([sys.executable,'-u',str(r/'code/audit_cache.py')],stdout=log,stderr=subprocess.STDOUT)
steps.append({'script':'audit_cache.py','exit_code':audit.returncode})
if audit.returncode:
    dump(r/'matched_pipeline_result.json',{'status':'failed','steps':steps})
    status('failed_cache_audit')
    sys.exit(audit.returncode)
for arm in read(r/'training_plan_v2.json')['arms']:
    existing = r/'training'/arm/'training_result.json'
    if existing.exists():
        assert read(existing)['ok']
        steps.append({'script':'train_arm_v4.py', 'arm':arm, 'reused_complete':True})
        continue
    gpu_used = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert gpu_used < 500, ('GPU occupied before training arm', arm, gpu_used)
    status('training', arm=arm)
    with (r/f'training_{arm}.log').open('ab') as log:
        p = subprocess.run([sys.executable,'-u',str(r/'code/train_arm_v4.py'),'--arm',arm],stdout=log,stderr=subprocess.STDOUT)
    steps.append({'script':'train_arm_v4.py','arm':arm,'exit_code':p.returncode})
    if p.returncode:
        dump(r/'matched_pipeline_result.json',{'status':'failed','steps':steps})
        status('failed', arm=arm)
        sys.exit(p.returncode)
status('evaluating_selected')
with (r/'evaluation.log').open('ab') as log:
    p = subprocess.run([sys.executable,'-u',str(r/'code/evaluate_selected.py')],stdout=log,stderr=subprocess.STDOUT)
steps.append({'script':'evaluate_selected.py','exit_code':p.returncode})
dump(r/'matched_pipeline_result.json',{'status':'complete' if p.returncode==0 else 'failed','steps':steps})
status('complete' if p.returncode==0 else 'failed')
sys.exit(p.returncode)
