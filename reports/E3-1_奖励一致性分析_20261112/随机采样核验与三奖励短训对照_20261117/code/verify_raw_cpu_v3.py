"""Independent CPU reconstruction from archived raw states/scores/traces; no training/eval helpers."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch
from tokenizers import Tokenizer
r=Path(sys.argv[1]).resolve();torch.set_num_threads(4)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def check(x,y,tol=1e-12):assert abs(float(x)-float(y))<=tol,(x,y,tol)
tokenizer=Tokenizer.from_file(str(r/'inputs/tokenizer_snapshot.json'));p=read(r/'plan.json');case={x['image']:x for x in read(r/'inputs/quantity_cases.json')};cohort={x['image']:x for x in read(r/'inputs/cohort.json')};gallery=read(r/'inputs/quantity_gallery.json');asset=torch.load(r/'assets/scoring_assets.pt',map_location='cpu',weights_only=True);qtext=asset['quantity_text_features'];proxy=asset['proxy_train_features'];words={w:i+1 for i,w in enumerate('one two three four five six seven'.split())};count=0;rounds=0;witnesses=0;all_rows=[];errors={'quantity_dot_max':0.,'proxy_dot_max':0.};sources=[]
for n in ['pilot/records.json']:
 if (r/n).exists():sources.append(n)
for policy in p['evaluation']['policies']:
 if (r/'evaluation'/policy/'records.json').exists():sources.append('evaluation/'+policy+'/records.json')
for folder in ['training','training_v2']:
 for arm in p['reward_arms']:
  path=r/folder/arm/'samples.jsonl'
  if path.exists():all_rows.extend(json.loads(x) for x in path.read_text().splitlines())
for n in sources:all_rows.extend(read(r/n))
for row in all_rows:
 assert sha(r/row['tensor_file'])==row['tensor_sha256'] and sha(r/row['trace_file'])==row['trace_sha256'];t=torch.load(r/row['tensor_file'],map_location='cpu',weights_only=True);tr=read(r/row['trace_file']);h=t['prior'];assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all();assert torch.equal(h,t['step_states'][-1][-1]) and digest(h)==tr['state_tensor_sha256'];zs=tr['rounds'];assert len(zs)==len(t['step_states']);events=tr['tool_events'];result=any('<result>' in z['generated_text'] or '</result>' in z['generated_text'] for z in zs);answer=zs[-1]['generated_text'];m=re.fullmatch(r'<answer>(.*?)</answer>',answer.strip(),re.S);strict=bool(tr['stop_reason']=='answer' and not result and m and m.group(1).strip() and answer.count('<answer>')==answer.count('</answer>')==1 and '<tool_call>' not in answer and '</tool_call>' not in answer and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in zs) and all(re.fullmatch(r'<tool_call>\s*scene_classify\s*</tool_call>',z['generated_text'].strip()) for z in zs[:-1]));consumed=any(e['after_round']<len(zs) for e in events);usable=strict and consumed and tr['tools_used']>0;assert usable==row['strict_completed_consumed_tool'] and consumed==row['feedback_consumed'] and result==row['generated_result_tag'];fmt=.2 if usable else 0.;check(row['r_format'],fmt)
 for j,(z,hs) in enumerate(zip(zs,t['step_states'])):
  ids=z['generated_ids'];assert tokenizer.decode(ids,skip_special_tokens=True)==z['generated_text'];assert [tokenizer.id_to_token(i) for i in ids]==z['generated_token_strings'];assert hs.shape==(len(ids),3584) and hs.dtype==torch.bfloat16 and torch.isfinite(hs).all() and len(ids)==z['generated_token_count'];assert digest(hs)==z['step_state_numeric_sha256'] and digest(hs[-1])==z['state_tensor_sha256'];assert len(z['prompt_token_ids'])>0 and ids[0] not in z['eos_token_ids'];hits=[i for i,x in enumerate(ids) if x in z['eos_token_ids']];assert not hits or hits==[len(ids)-1];rounds+=1
  if not row.get('reuse_source'):assert len(z['actual_generation_token_logprobs'])==len(ids) and all(np.isfinite(z['actual_generation_token_logprobs']))
 for ri,w in t.get('probability_witness',{}).items():
  logits=w['generation_logits'].double();lp=torch.log_softmax(logits,dim=-1).gather(1,w['targets'][:,None]).flatten();actual=torch.tensor(zs[int(ri)]['actual_generation_token_logprobs'])[w['positions']];assert torch.max(torch.abs(lp-actual))<2e-5;witnesses+=1
 qs=t['quantity_scores'];dot=(t['image_feature']@qtext.T).flatten();err=float((dot-qs).abs().max());errors['quantity_dot_max']=max(errors['quantity_dot_max'],err);assert err<1e-6
 if row['split']=='train':
  target=row['row'];assert asset['proxy_train_names'][target]==row['image'];scores=t['proxy_scores'];err=float(((t['proxy_feature']@proxy.T).flatten()-scores).abs().max());errors['proxy_dot_max']=max(errors['proxy_dot_max'],err);assert err<1e-6;rank=int((scores>scores[target]).sum());assert rank==row['proxy_rank0'];check(row['r_proxy_rank'],fmt+1/(1+rank))
 else:assert row['r_proxy_rank'] is None
 if row['image'] in case:
  c=case[row['image']]
  for word in ['A','B']:
   pos,neg=gallery['quantity_columns'][row['image']][word];prefix=gallery['quantity_text_order'][pos].rsplit(' ',1)[0];wrong=[i for i,s in enumerate(gallery['quantity_text_order']) if s.rsplit(' ',1)[0]==prefix and words[s.rsplit(' ',1)[1][:-1]]!=c['actual_count']];pair=float(qs[pos])-float(qs[neg]);hard=float(qs[pos])-max(float(qs[i]) for i in wrong);check(row[word+'_pair_margin'],pair);check(row[word+'_hard_margin'],hard);assert row[word+'_pair_correct']==(pair>0) and row[word+'_hard_correct']==(hard>0)
   if word=='A':check(row['r_quantity_pair'],fmt+pair);check(row['r_quantity_hard'],fmt+hard)
 count+=1
prob={}
for group in ['probability_gate','probability_gate_v2']:
 if (r/group/'result.json').exists():
  gate=read(r/group/'result.json');assert gate['ok'] and gate['permanent_Agent_updates']==0 and gate['restored_policy_hash']==p['policy']['lora_hash']==gate['initial_policy_hash'];d=read(r/group/'rounds.json');w=torch.load(r/group/'probability_witness.pt',map_location='cpu',weights_only=True)
  for z in d:assert z['max_token_logprob_gap']<=.25 and z['mean_absolute_token_logprob_gap']<=.05 and z['last_hidden_cosine']>=.999
  for key,x in w.items():
   teacher=torch.log_softmax(x['teacher_logits'].double(),dim=-1).gather(1,x['targets'][:,None]).flatten();a,b=map(int,key.split('_'));record=next(z for z in d if z['sample']==a and z['round']==b);assert torch.max(torch.abs(teacher-torch.tensor(record['token_logprobs'])[x['positions']]))<2e-5
  prob[group]={'bounded_gate_samples':gate['checked_samples'],'bounded_gate_rounds':len(d),'temporary_update_restored':True}
result={'ok':True,'records_rebuilt':count,'rounds_rebuilt':rounds,'generation_full_vocabulary_witness_rows':witnesses,'max_CPU_vs_GPU_arithmetic_errors':errors,'probability_gate':prob,'torch':torch.__version__,'numpy':np.__version__};path=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else r/'independent_raw_verification.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
