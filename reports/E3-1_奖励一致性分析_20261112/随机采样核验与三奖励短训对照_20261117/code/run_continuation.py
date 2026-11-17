"""Sealed gated queue: wait pilot, probability witness, common-start3arms, fixed4policy eval."""
import os
from pathlib import Path
import subprocess
import sys
import time
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent
for manifest in ['pilot_code_hashes_v2.json','continuation_code_hashes.json']:
 for n,h in read(r/manifest).items():assert sha(r/n)==h,n
assert not (r/'continuation_job_result.json').exists()
while not (r/'initial_job_result_v2.json').exists():time.sleep(5)
initial=read(r/'initial_job_result_v2.json')
if initial['status']!='complete':dump(r/'continuation_job_result.json',{'status':'preparation_failed','source':initial});raise SystemExit(1)
lock=read(r/'prompt_selection_lock.json')
if lock['selected_prompt'] is None:dump(r/'continuation_job_result.json',{'status':'pilot_protocol_gate_failed','Agent_optimizer_updates':0,'readout_optimizer_updates':0,'training_arms_completed':0});raise SystemExit(0)
jobs=[['probability_gate.py'],['baseline_replay.py']]+[['train_reward_arm.py',arm] for arm in read(r/'plan.json')['reward_arms']]+[['evaluate_policy.py',name] for name in read(r/'plan.json')['evaluation']['policies']]
for args in jobs:
 if (r/'PAUSE_REQUESTED').exists():dump(r/'continuation_job_result.json',{'status':'paused','next_entry':args});raise SystemExit(75)
 logfile=r/('queue_'+ '_'.join(args).replace('.py','')+'.log')
 with logfile.open('xb') as f:p=subprocess.run([sys.executable,'-u',str(r/'code'/args[0]),*args[1:]],cwd=r,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONHASHSEED':'0'})
 if p.returncode:
  dump(r/'continuation_job_result.json',{'status':'paused' if p.returncode==75 else 'failed','entry':args,'exit_code':p.returncode});raise SystemExit(p.returncode)
dump(r/'continuation_job_result.json',{'status':'generation_training_evaluation_complete','arms':{k:read(r/'training'/k/'result.json') for k in read(r/'plan.json')['reward_arms']},'policies':{k:read(r/'evaluation'/k/'result.json') for k in read(r/'plan.json')['evaluation']['policies']},'analysis_and_audit':'pending'})
