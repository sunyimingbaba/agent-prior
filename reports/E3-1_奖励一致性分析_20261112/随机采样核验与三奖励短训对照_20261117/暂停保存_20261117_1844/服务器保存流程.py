from pathlib import Path
import json,hashlib,subprocess,os,shutil,time,datetime
import torch
from safetensors.torch import load_file
r=Path('/private/AGSE_E3_1_rebuild_20261112/rfpo_sampling_reward_comparison_20261117');base=Path('/private/AGSE_E3_1_rebuild_20261112/rfpo_sampling_reward_pause_20261117_1844');torch.set_num_threads(4)
for _ in range(12):
 if (r/'post_job_result.json').exists():break
 time.sleep(5)
assert json.loads((r/'continuation_job_result_v2.json').read_text())['status']=='paused';assert json.loads((r/'post_job_result.json').read_text())['status']=='prior_stage_stopped'
m=json.loads((r/'training_v2/quantity_pair/paused_12_checkpoint_manifest.json').read_text());cp=r/'training_v2/quantity_pair/paused_12/checkpoint'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
for n,v in m['files'].items():assert (cp/n).stat().st_size==v['bytes'] and sha(cp/n)==v['sha256'],n
w=load_file(str(cp/'adapter_model.safetensors'));live={n.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight'):v for n,v in w.items()};h=hashlib.sha256()
for n,v in sorted(live.items()):assert v.dtype==torch.float32 and torch.isfinite(v).all();h.update(n.encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
assert len(live)==584 and h.hexdigest()==m['policy_hash'];opt=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=True,mmap=True);assert len(opt['state'])==584 and {int(v['step']) for v in opt['state'].values()}=={12} and all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in opt['state'].values());us=[json.loads(v) for v in (r/'training_v2/quantity_pair/updates.jsonl').read_text().splitlines()];assert len(us)==12 and sum(x['optimizer_update'] for x in us)==12 and us[-1]['policy_hash_after']==m['policy_hash'];assert all((cp/n).is_file() for n in ['torch_rng.pt','python_numpy_rng.json','training_state.json']);proof={'ok':True,'arm':'quantity_pair','attempts_completed':12,'actual_optimizer_updates':12,'checkpoint_files_rehashed':len(m['files']),'LoRA_tensors':584,'all_Adam_state_steps':12,'policy_hash':h.hexdigest(),'RNG_state_files_bound':True};(r/'pause_checkpoint_verification.json').write_text(json.dumps(proof,indent=2));print('PAUSED_CHECKPOINT_VERIFIED',flush=True)
plan=json.loads((r/'plan.json').read_text());summary={'status':'paused_by_user','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proxy_rank':json.loads((r/'training_v2/proxy_rank/result.json').read_text()),'quantity_pair':{**proof,'checkpoint':'training_v2/quantity_pair/paused_12/checkpoint','next_attempt':13,'next_image':plan['training']['image_order'][12],'next_sample_seed0':2026111800+100*12},'quantity_hard':{'attempts_completed':0,'optimizer_updates':0},'formal_candidate_records':256+96,'pilot_records':128,'zero_update_failure_records':8,'permanent_optimizer_updates_this_comparison':31+12,'evaluation_records':0,'model_readout_optimizer_updates':0,'resume_rule':'restore exact paused LoRA+AdamW+CPU/CUDA/Python/NumPy RNG, continueattempt13/32; preserve43existingpermanentupdates and allpriorrecords; training must wait for explicituserresume'};(r/'pause_summary_20261117.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));ps=subprocess.check_output(['ps','-eo','pid,args'],text=True);assert not any(str(r/'code/train_reward_arm_v2.py') in line or str(r/'code/run_continuation_v2.py') in line or str(r/'code/run_post_queue_v2.py') in line for line in ps.splitlines());raw=base/'raw';raw.mkdir();files=[]
for p in r.rglob('*'):
 rel=p.relative_to(r)
 if any(x in rel.parts for x in ['scratch','__pycache__']) or (len(rel.parts)==1 and p.name.endswith('.tar.gz')):continue
 assert not p.is_symlink(),str(rel)
 if p.is_file():
  dest=raw/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  try:os.link(p,dest)
  except OSError:shutil.copy2(p,dest)
  files.append(str(rel))
assert not (raw/'file_checksums.json').exists();print('QUIESCENT_SNAPSHOT_CREATED',len(files),flush=True);p=subprocess.run(['/private/AGSE_E1_20261110/legacy/env/bin/python',str(r/'code/package_complete_evidence.py'),str(raw),str(base/'rfpo_sampling_reward_pause_20261117_1844.tar.gz')],text=True);assert p.returncode==0;print('PAUSE_ARCHIVE_COMPLETE',flush=True)
