"""One reward arm: common start/seeds, bounded32attempts, exact saved/scored generated states."""
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import numpy as np
from milestone_common import ROOT,setup,seed,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from stochastic_rollout import rollout
from prompt_design import messages
from policy_logprob import grad_logprob
from reward_model_v2 import RewardScorer
from stage_io import read,dump,sha
arm=sys.argv[1];torch=setup();p=read(ROOT/'plan.json');cfg=p['training'];assert arm in p['reward_arms'];assert read(ROOT/'probability_gate/result.json')['ok'];lock=read(ROOT/'prompt_selection_lock.json');prompt=lock['selected_prompt'];assert prompt
out=ROOT/'training'/arm;out.mkdir(parents=True);start=time.monotonic();seed(0);model,processor=load_policy(Path(p['policy']['checkpoint']),True);restore_adapter_exact(model,Path(p['policy']['checkpoint']));initial=lora_hash(model);assert initial==p['policy']['lora_hash'];previous=initial
scorer=RewardScorer();by={x['image']:x for x in read(ROOT/'inputs/cohort.json')};counts={x['image']:x['actual_count'] for x in read(ROOT/'inputs/quantity_cases.json')};parameters=[v for v in model.parameters() if v.requires_grad];assert len(parameters)==584 and all(v.dtype==torch.float32 for v in parameters)
opt=torch.optim.AdamW(parameters,lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay']);updates=[];records=[];actual=0;pause=False
from agent.rollout import SYSTEM_PROMPT
def want_pause(signum,frame):
 global pause;pause=True
signal.signal(signal.SIGUSR1,want_pause)
def append(path,value):
 with path.open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def checkpoint(label,attempts):
 folder=out/label/'checkpoint';folder.mkdir(parents=True);model.save_pretrained(folder,safe_serialization=True);processor.save_pretrained(folder);torch.save(opt.state_dict(),folder/'optimizer.pt');torch.save({'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()},folder/'torch_rng.pt');ns=np.random.get_state();dump(folder/'python_numpy_rng.json',{'python_random_state':random.getstate(),'numpy_random_state':{'kind':ns[0],'keys':ns[1].tolist(),'position':ns[2],'has_gauss':ns[3],'cached_gaussian':ns[4]}});dump(folder/'training_state.json',{'arm':arm,'attempts_completed':attempts,'optimizer_updates':actual,'policy_hash':lora_hash(model),'initial_policy_hash':initial,'plan_sha256':sha(ROOT/'plan.json'),'prompt_lock_sha256':sha(ROOT/'prompt_selection_lock.json')});manifest={'arm':arm,'attempts_completed':attempts,'optimizer_updates':actual,'policy_hash':lora_hash(model),'files':{str(f.relative_to(folder)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in folder.rglob('*') if f.is_file()}};dump(out/(label+'_checkpoint_manifest.json'),manifest);return str(folder.relative_to(ROOT))
dump(out/'settings.json',{'arm':arm,'reward_formula':p['reward_arms'][arm],'config':cfg,'initial_policy_hash':initial,'readout_frozen':scorer.verify_frozen(),'prompt':prompt,'trainable_tensors':len(parameters),'trainable_parameters':sum(v.numel() for v in parameters),'seed_schedule':cfg['sample_seed'],'fresh_optimizer':True,'zero_variance_policy':cfg['zero_variance_policy']})
try:
 for ai,name in enumerate(cfg['image_order']):
  c=by[name];image=scorer.case_image(c);group=[]
  for sample in range(cfg['group_size']):
   ss=2026111800+100*ai+sample;seed(ss);h,tensors,tr=rollout(model,processor,image,messages(prompt,c,SYSTEM_PROMPT),{**c['feedback'],'source_image':name,'source_original_trace_sha256':c['feedback_original_trace_sha256']},p['rollout'],temperature=1.)
   values=scorer.score(h,c,tr['strict_completed_consumed_tool']);folder=out/'samples';folder.mkdir(exist_ok=True);stem=f'{ai+1:02d}_{c["key"]}_{sample:02d}';sf=folder/(stem+'.pt');tf=folder/(stem+'.json');torch.save({**tensors,**scorer.last_tensors},sf);dump(tf,tr)
   row={'id':stem,'arm':arm,'attempt':ai+1,'image':name,'key':c['key'],'split':'train','row':c['row'],'actual_count':counts[name],'sample':sample,'seed':ss,'policy_hash_before':previous,'strict_completed_consumed_tool':tr['strict_completed_consumed_tool'],'feedback_consumed':tr['feedback_consumed'],'tools_used':tr['tools_used'],'generated_result_tag':tr['generated_result_tag'],'stop_reason':tr['stop_reason'],**values,'reward':values['r_'+arm],'tensor_file':str(sf.relative_to(ROOT)),'tensor_sha256':sha(sf),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf)};append(out/'samples.jsonl',row);records.append(row);group.append((row,tr))
  rewards=torch.tensor([x['reward'] for x,_ in group],device='cuda',dtype=torch.float32);std=rewards.std(unbiased=True);informative=float(std)>1e-8;advantages=((rewards-rewards.mean())/(std+1e-8)).detach() if informative else torch.zeros_like(rewards);opt.zero_grad(set_to_none=True);model.train();loss_sum=0.;tokens=0;prob=[];norm=None
  if informative:
   for sample,(row,tr) in enumerate(group):
    for ri,z in enumerate(tr['rounds']):
     lp,d,_=grad_logprob(model,processor,image,z);delta=torch.tensor(d['token_logprobs'])-torch.tensor(z['actual_generation_token_logprobs']);mx=float(delta.abs().max());mean=float(delta.abs().mean());e={'sample':sample,'round':ri,'max_token_logprob_gap':mx,'mean_absolute_token_logprob_gap':mean,'advantage':float(advantages[sample]),**d};prob.append(e);assert mx<=p['probability_gate']['max_token_gap'] and mean<=p['probability_gate']['mean_absolute_gap'],'on-policy likelihood replay threshold'
     loss=-advantages[sample]*lp/cfg['group_size'];assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach());tokens+=d['effective_tokens'];del lp,loss
   norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip_norm']));assert math.isfinite(norm) and norm>0 and all(v.grad is None or torch.isfinite(v.grad).all() for v in parameters) and all(v.grad is None for n,v in model.named_parameters() if 'lora_' not in n)
   opt.step();actual+=1;current=lora_hash(model);assert current!=previous and all(torch.isfinite(v).all() for v in parameters)
  else:current=lora_hash(model);assert current==previous
  scorer.verify_offset();assert all(v.grad is None and not v.requires_grad for v in scorer.model.parameters())
  update={'arm':arm,'attempt':ai+1,'image':name,'actual_count':counts[name],'optimizer_update':informative,'optimizer_updates_so_far':actual,'reward_mean':float(rewards.mean()),'reward_std':float(std),'advantages':advantages.cpu().tolist(),'loss':loss_sum,'effective_loss_tokens':tokens,'gradient_norm_before_clip':norm,'policy_hash_before':previous,'policy_hash_after':current,'strict_completed_count':sum(x['strict_completed_consumed_tool'] for x,_ in group),'generated_result_count':sum(x['generated_result_tag'] for x,_ in group),'probability_replays':prob};append(out/'updates.jsonl',update);updates.append(update);previous=current;dump(ROOT/'training_progress.json',{'arm':arm,'attempts_completed':ai+1,'attempts_total':32,'optimizer_updates':actual,'elapsed_seconds':time.monotonic()-start});print('TRAIN_ARM',arm,ai+1,'/32','actual_updates',actual,'strict',update['strict_completed_count'],'/8',flush=True)
  if ai+1==16:checkpoint('intermediate_16',16)
  if pause or (ROOT/'PAUSE_REQUESTED').exists():checkpoint('paused_'+str(ai+1),ai+1);dump(out/'paused.json',{'attempts_completed':ai+1,'optimizer_updates':actual});sys.exit(75)
 assert len(records)==256 and len(updates)==32;path=checkpoint('final',32);dump(out/'result.json',{'ok':True,'arm':arm,'attempts':32,'optimizer_updates':actual,'zero_variance_skips':32-actual,'samples':256,'training_images':8,'initial_policy_hash':initial,'final_policy_hash':previous,'checkpoint':path,'fixed_checkpoint_selection':'final32attempts','strict_completed_samples':sum(x['strict_completed_consumed_tool'] for x in records),'readout_frozen':scorer.verify_frozen(),'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});print('ARM_COMPLETE',arm,actual,flush=True)
except Exception:
 dump(out/'failure.json',{'error':traceback.format_exc(),'attempts_completed':len(updates),'samples_saved':len(records),'optimizer_updates':actual});checkpoint('interrupted_'+str(len(updates)),len(updates));raise
