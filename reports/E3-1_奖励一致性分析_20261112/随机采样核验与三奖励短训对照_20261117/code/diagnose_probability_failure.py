"""Bounded diagnosis on the failed zero-update group, before any revised training decision."""
from pathlib import Path
import json
import time
from milestone_common import ROOT,setup,seed,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from policy_logprob import grad_logprob as full
from policy_logprob_cached import grad_logprob as cached
from reward_model_v2 import RewardScorer
from stage_io import read,dump,sha
torch=setup();p=read(ROOT/'plan.json');failure=read(ROOT/'training/proxy_rank/failure.json');assert failure['optimizer_updates']==0;out=ROOT/'probability_failure_diagnostic';out.mkdir();start=time.monotonic();seed(0);model,processor=load_policy(Path(p['policy']['checkpoint']),True);restore_adapter_exact(model,Path(p['policy']['checkpoint']));assert lora_hash(model)==p['policy']['lora_hash'];scorer=RewardScorer();by={x['image']:x for x in read(ROOT/'inputs/cohort.json')};samples=[json.loads(v) for v in (ROOT/'training/proxy_rank/samples.jsonl').read_text().splitlines()];assert len(samples)==8;rows=[];bad=[]
for row in samples:
 tr=read(ROOT/row['trace_file']);image=scorer.case_image(by[row['image']])
 for ri,z in enumerate(tr['rounds']):
  with torch.no_grad():lp,d,_=full(model,processor,image,z)
  delta=torch.tensor(d['token_logprobs'])-torch.tensor(z['actual_generation_token_logprobs']);mx=float(delta.abs().max());mean=float(delta.abs().mean());item={'sample':row['sample'],'round':ri,'tokens':len(z['generated_ids']),'max_full_token_gap':mx,'mean_full_abs_gap':mean};rows.append(item)
  if mx>.25 or mean>.05:bad.append((row,ri,z,item))
  del lp,d
 dump(out/'full_replay_rounds.json',rows)
 print('FAILED_GROUP_FULL_REPLAY',row['sample'],flush=True)
assert bad,'expected reproduced guard failure';row,ri,z,item=bad[0];image=scorer.case_image(by[row['image']]);model.zero_grad(set_to_none=True);lp,d,x=cached(model,processor,image,z,True);delta=torch.tensor(d['token_logprobs'])-torch.tensor(z['actual_generation_token_logprobs']);mx=float(delta.abs().max());mean=float(delta.abs().mean());(-lp).backward();params=[v for v in model.parameters() if v.requires_grad];norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params);before=p['policy']['lora_hash'];assert lora_hash(model)==before;torch.save(x,out/'cached_witness.pt');dump(out/'cached_replay.json',{'sample':row['sample'],'round':ri,'trace_file':row['trace_file'],'trace_sha256':row['trace_sha256'],'full_replay':item,'cached_max_token_gap':mx,'cached_mean_absolute_gap':mean,'cached_token_logprobs':d['token_logprobs'],'gradient_norm_before_clip':norm,'finite_gradients':True,'policy_hash_before_and_after':before,'Agent_optimizer_updates':0,'readout_frozen':scorer.verify_frozen(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-start,'same_original_gate_passes':mx<=.25 and mean<=.05});print('CACHED_DIAGNOSTIC',mx,mean,norm,flush=True)
