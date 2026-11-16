"""Adopt the uninterrupted training child and normalize legacy checkpoint counters."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,dump,sha

r=Path(__file__).resolve().parent.parent
adoption=read(r/'queue_adoption.json')
steps=[]
def status(phase,**extra):
    dump(r/'matched_pipeline_progress.json',{'phase':phase,'steps':steps,'orchestrator':'v2',
         'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'
def gpu_free():
    for _ in range(60):
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used<500:return
        time.sleep(2)
    raise RuntimeError('GPU remained occupied after the owned training child completed')
for manifest in ['training_code_hashes_v4.json','evaluation_code_hashes_v2.json','queue_code_hashes_v2.json']:
    for n,h in read(r/manifest).items():assert sha(r/n)==h,n
assert read(r/'cache_audit.json')['ok']
for arm in read(r/'training_plan_v2.json')['arms']:
    result=r/'training'/arm/'training_result.json'
    if arm==adoption['adopted_arm'] and not result.exists():
        status('training',arm=arm,adopted_pid=adoption['adopted_training_pid'])
        while not result.exists():
            if not alive(adoption['adopted_training_pid']):
                dump(r/'matched_pipeline_result_v2.json',{'status':'failed','reason':'adopted training child exited without result','steps':steps})
                sys.exit(1)
            time.sleep(5)
        assert read(result)['ok']
        steps.append({'script':'train_arm_v4.py','arm':arm,'adopted_uninterrupted':True,
                      'pid':adoption['adopted_training_pid'],'exit_code':0})
        continue
    if result.exists():
        assert read(result)['ok']
        steps.append({'script':'train_arm_v4.py','arm':arm,'reused_complete':True,'exit_code':0})
        continue
    gpu_free();status('training',arm=arm)
    with (r/f'training_{arm}.log').open('ab') as log:
        p=subprocess.run([sys.executable,'-u',str(r/'code/train_arm_v4.py'),'--arm',arm],stdout=log,stderr=subprocess.STDOUT)
    steps.append({'script':'train_arm_v4.py','arm':arm,'exit_code':p.returncode})
    if p.returncode:
        dump(r/'matched_pipeline_result_v2.json',{'status':'failed','steps':steps});status('failed',arm=arm);sys.exit(p.returncode)
gpu_free();status('evaluating_selected')
with (r/'evaluation_v2.log').open('ab') as log:
    p=subprocess.run([sys.executable,'-u',str(r/'code/evaluate_selected_v2.py')],stdout=log,stderr=subprocess.STDOUT)
steps.append({'script':'evaluate_selected_v2.py','exit_code':p.returncode})
dump(r/'matched_pipeline_result_v2.json',{'status':'complete' if p.returncode==0 else 'failed','steps':steps,
     'all_training_arms_same_v4_entry':True,'native_training_uninterrupted':True})
status('complete' if p.returncode==0 else 'failed');sys.exit(p.returncode)
