"""Actual sampled-token probability/backward witness; temporary update restored before continuation."""
import math
from pathlib import Path
import traceback
from milestone_common import ROOT,setup,seed,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from policy_logprob import grad_logprob
from reward_model_v2 import RewardScorer
from stage_io import read,dump,sha
torch=setup();p=read(ROOT/'plan.json');lock=read(ROOT/'prompt_selection_lock.json');assert lock['selected_prompt'] is not None
out=ROOT/'probability_gate';out.mkdir();start=p['policy']['lora_hash'];records=read(ROOT/'pilot/records.json');rows=[x for x in records if x['prompt']==lock['selected_prompt']][:2];by={c['image']:c for c in read(ROOT/'inputs/cohort.json')}
seed(0);model,processor=load_policy(Path(p['policy']['checkpoint']),True);restore_adapter_exact(model,Path(p['policy']['checkpoint']));assert lora_hash(model)==start
scorer=RewardScorer();params=[v for v in model.parameters() if v.requires_grad];assert len(params)==584 and all(v.dtype==torch.float32 for v in params)
cfg=p['training'];opt=torch.optim.AdamW(params,lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay']);model.train();opt.zero_grad(set_to_none=True);details=[];witness={};ok=False
try:
 for j,row in enumerate(rows):
  tr=read(ROOT/row['trace_file']);saved=torch.load(ROOT/row['tensor_file'],map_location='cpu',weights_only=True);image=scorer.case_image(by[row['image']])
  for ri,z in enumerate(tr['rounds']):
   lp,d,x=grad_logprob(model,processor,image,z,True);delta=torch.tensor(d['token_logprobs'])-torch.tensor(z['actual_generation_token_logprobs']);mx=float(delta.abs().max());mean=float(delta.abs().mean());cos=float(torch.nn.functional.cosine_similarity(x['teacher_hidden'].float(),saved['step_states'][ri][-1].float(),dim=0))
   item={'sample':j,'round':ri,'image':row['image'],'trace_sha256':row['trace_sha256'],'max_token_logprob_gap':mx,'mean_absolute_token_logprob_gap':mean,'last_hidden_cosine':cos,**d};details.append(item)
   x['targets']=torch.tensor(z['generated_ids'])[x['positions']];gen=saved['probability_witness'].get(ri)
   if gen is not None:x['generation_logits']=gen['generation_logits'];assert gen['positions']==x['positions'] and torch.equal(gen['targets'],x['targets'])
   witness[f'{j}_{ri}']=x
   assert mx<=p['probability_gate']['max_token_gap'] and mean<=p['probability_gate']['mean_absolute_gap'] and cos>=.999,'probability/hidden replay gate'
   loss=-lp/(len(rows)*len(tr['rounds']));assert torch.isfinite(loss);loss.backward();del lp,loss
  print('PROBABILITY_WITNESS',j+1,'/',len(rows),flush=True)
 norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip_norm']));assert math.isfinite(norm) and norm>0 and all(v.grad is None or torch.isfinite(v.grad).all() for v in params) and all(v.grad is None for n,v in model.named_parameters() if 'lora_' not in n)
 opt.step();temporary=lora_hash(model);assert temporary!=start and all(torch.isfinite(v).all() for v in params);scorer.verify_frozen();ok=True
except Exception:
 error=traceback.format_exc();dump(out/'failure.json',{'error':error,'rounds':details});print(error,flush=True)
finally:
 restore_adapter_exact(model,Path(p['policy']['checkpoint']));restored=lora_hash(model);assert restored==start;opt.zero_grad(set_to_none=True);torch.save(witness,out/'probability_witness.pt');dump(out/'rounds.json',details)
 dump(out/'result.json',{'ok':ok,'permanent_Agent_updates':0,'temporary_updates':int(ok),'initial_policy_hash':start,'restored_policy_hash':restored,'restored_exact':True,'temporary_policy_hash':temporary if ok else None,'gradient_norm_before_clip':norm if ok else None,'checked_samples':len(rows),'checked_rounds':len(details),'trainable_tensors':len(params),'trainable_parameters':sum(v.numel() for v in params),'readout_frozen':scorer.verify_frozen(),'witness_sha256':sha(out/'probability_witness.pt')})
assert ok,'probability gate stopped continuation'
print('PROBABILITY_GATE_PASSED_RESTORED',flush=True)
