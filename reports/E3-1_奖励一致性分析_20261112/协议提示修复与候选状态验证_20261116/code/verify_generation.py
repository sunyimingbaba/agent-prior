"""Independent native-state, actual-token byte alignment and train-only protocol gate audit."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--matched',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numeric(t):return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
d=read(r/'protocol_plan.json');records=read(r/'generation_v2/records.json');cohort=read(r/'inputs/cohort.json');bykey={c['key']:c for c in cohort};byrecord={(x['group'],x['key']):x for x in records};tok=Tokenizer.from_file(str(r/'inputs/tokenizer_snapshot.json'))
assert len(byrecord)==len(records)
for name in ['generation_code_hashes.json','generation_code_hashes_v2.json','evaluation_code_hashes.json','post_code_hashes.json']:
 for n,h in read(r/name).items():assert sha(r/n)==h,n
allowed=set(range(33,127))|set(range(161,173))|set(range(174,256));missing=[i for i in range(256) if i not in allowed]
def byte_token(s):return bytes(ord(c) if ord(c) in allowed else missing[ord(c)-256] for c in s)
def layout(step):
 ids=step['generated_ids'];tokens=step['generated_token_strings'];assert [tok.id_to_token(i) for i in ids]==tokens
 eos=step['eos_token_ids'];end=[i for i,x in enumerate(ids) if x in eos]
 if end and end!=[len(ids)-1]:return None
 n=len(ids)-int(bool(end))
 try:parts=[byte_token(t) for t in tokens[:n]];raw=b''.join(parts);text=raw.decode('utf8')
 except (IndexError,UnicodeDecodeError):return None
 if text!=step['generated_text'] or text.count('<answer>')!=1 or text.count('</answer>')!=1 or not re.fullmatch(r'<answer>\s*.+?\s*</answer>',text.strip(),re.S):return None
 start=raw.index(b'<answer>')+8;end=raw.index(b'</answer>');offset=0;use=[];spans=[]
 for i,b in enumerate(parts):
  lo,hi=offset,offset+len(b);spans.append([lo,hi]);offset=hi
  if lo>=start and hi<=end and hi>lo and i+1<len(ids):use.append(i)
 if not use:return None
 return {'byte_span':[start,end],'token_indices':use,'step_indices':[i+1 for i in use],'spans':spans}
groups={};states_verified=step_values=body_positions=0;dangling=0
for x in records:
 c=bykey[x['key']];assert (x['image'],x['split'],x['row'])==(c['image'],c['split'],c['row'])
 assert sha(r/x['tensor_file'])==x['tensor_sha256'] and sha(r/x['trace_file'])==x['trace_sha256'];saved=torch.load(r/x['tensor_file'],map_location='cpu',weights_only=True);tr=read(r/x['trace_file']);hs=saved['step_states'];rounds=tr['rounds'];assert len(hs)==len(rounds)<=3
 generated_result=any('<result>' in q['generated_text'] or '</result>' in q['generated_text'] for q in rounds);assert generated_result==x['generated_result_tag']==tr['generated_result_tag']
 for q,h in zip(rounds,hs):
  assert h.shape==(len(q['generated_ids']),3584) and h.dtype==torch.bfloat16 and torch.isfinite(h).all();assert numeric(h)==q['step_state_numeric_sha256'] and numeric(h[-1])==q['state_tensor_sha256']
  assert hashlib.sha256(q['prompt_text'].encode()).hexdigest()==q['prompt_text_sha256'];assert q['generated_token_count']==h.shape[0]<=256;step_values+=h.shape[0]
  assert [tok.id_to_token(i) for i in q['generated_ids']]==q['generated_token_strings']
 final=rounds[-1]['generated_text'];match=re.fullmatch(r'<answer>(.*?)</answer>',final.strip(),re.S);single=bool(final.count('<answer>')==1 and final.count('</answer>')==1 and match and match.group(1).strip() and '<tool_call>' not in final and '</tool_call>' not in final);tool_pattern=r'<tool_call>\s*scene_classify\s*</tool_call>'
 prior_calls=all(re.fullmatch(tool_pattern,q['generated_text'].strip()) for q in rounds[:-1]);strict=bool(single and prior_calls and not generated_result and all(not q['token_limit'] and 'addCriterion' not in q['generated_text'] for q in rounds))
 if tr['strict_protocol_usable'] and ('</tool_call>' in final or '<tool_call>' in final):dangling+=1;strict=False
 assert strict==x['strict_protocol_usable']==tr['strict_protocol_usable'],(x['group'],x['image'],'independent strict mismatch')
 assert len(tr['tool_events'])==x['tools_used'];consumed=any(e['after_round']<len(rounds) for e in tr['tool_events']);assert consumed==x['feedback_consumed']
 for event in tr['tool_events']:
  assert event['text']==c['feedback']['text'] and event['confidence']==c['feedback']['confidence'] and event['source_image']==c['image'] and event['source_original_trace_sha256']==c['feedback_original_trace_sha256']
  k=event['after_round'];assert re.fullmatch(tool_pattern,rounds[k-1]['generated_text'].strip())
  if k<len(rounds):assert '<result>'+event['text']+'</result>' in rounds[k]['prompt_text'] and rounds[k]['message_roles'][-2:]==['assistant','user']
 valid=bool(strict and consumed and len(rounds)>1);mapped=layout(rounds[-1]);pooled=bool(valid and mapped)
 expected={'last':hs[-1][-1],'answer_prefill':hs[-1][0] if valid else hs[-1][-1]}
 if pooled:
  matrix=hs[-1][mapped['step_indices']].double().numpy();mean=np.sum(matrix,axis=0,dtype=np.float64)/len(matrix);expected['answer_pool']=torch.from_numpy(mean).to(torch.bfloat16);body_positions+=len(matrix)
  m=rounds[-1]['body_layout'];assert m['body_byte_span']==mapped['byte_span'] and m['body_token_indices']==mapped['token_indices'] and m['body_state_step_indices']==mapped['step_indices'] and m['response_token_byte_spans']==mapped['spans']
 else:expected['answer_pool']=hs[-1][-1]
 assert x['coverage']['answer_prefill_available']==valid and x['coverage']['answer_pool_available']==pooled
 for state,h in expected.items():assert torch.equal(h,saved['candidates'][state]) and numeric(h)==x['candidate_numeric_sha256'][state];states_verified+=1
 if x['group']=='baseline':
  assert numeric(expected['last'])==c['numeric_state_sha256'];old=read(a.matched/c['trace_file']);assert [q['generated_ids'] for q in rounds]==[q['generated_ids'] for q in old['rounds']]
 groups.setdefault(x['group'],[]).append(x)
assert dangling==0
selection=read(r/'prompt_selection_lock_v2.json');result=read(r/'generation_result_v2.json');prefix='pilot_after_sft' if selection['Agent_optimizer_updates'] else 'pilot_initial';outcomes={}
for candidate in d['prompt_candidates_priority']:
 xs=groups[prefix+'/'+candidate];assert len(xs)==63 and all(x['split']=='train' for x in xs);v=sum(x['strict_protocol_usable'] and x['tools_used']>0 and x['feedback_consumed'] for x in xs);bad=sum(x['generated_result_tag'] for x in xs);passed=v/63>=.95 and bad/63<=.02;assert selection['train_only_outcomes'][candidate]['strict_completed_with_consumed_tool']==v and selection['train_only_outcomes'][candidate]['passed']==passed;outcomes[candidate]=passed
passing=[n for n in d['prompt_candidates_priority'] if outcomes[n]]
if passing:assert selection['chosen_prompt']==passing[0] and selection['protocol_gate_passed']
else:
 assert not selection['protocol_gate_passed'] and result['exploratory_after_failed_repair'];assert selection['chosen_prompt']==max(d['prompt_candidates_priority'],key=lambda n:(selection['train_only_outcomes'][n]['strict_completed_with_consumed_tool'],-d['prompt_candidates_priority'].index(n)))
assert result['Agent_optimizer_updates']==selection['Agent_optimizer_updates'] in [0,30] and result['readout_optimizer_updates']==0
if not result['Agent_optimizer_updates']:assert not (r/'training').exists() and result['initial_policy_hash']==result['final_policy_hash']==d['policy']['lora_hash']
for c in [x for x in cohort if x['split']=='train'][:2]:
 p=r/'generation_v2'/('anchor_'+c['key']+'.json');t=read(p);sf=r/t['anchor_tensor_file'];assert sha(sf)==t['anchor_tensor_sha256'];z=torch.load(sf,map_location='cpu',weights_only=True)
 assert numeric(z['candidates']['last'])==c['numeric_state_sha256'];assert [numeric(h) for h in z['step_states']]==[q['step_state_numeric_sha256'] for q in t['rounds']]
banks_verified=0
for f in d['state_families']:
 bank=torch.load(r/'state_banks'/(f+'.pt'),map_location='cpu',weights_only=True);meta=read(r/'state_banks'/(f+'.json'));state=f.removeprefix('legacy_').removeprefix('selected_')
 for split,n in [('train',63),('val',210)]:
  assert bank[split].shape==(n,3584) and len(meta[split])==n
  for i,x in enumerate(meta[split]):
   group='baseline' if f.startswith('legacy_') else (selection['selected_train_group'] if split=='train' else 'selected_validation');source=byrecord[group,x['key']];assert x['record_group']==group and x['candidate']==state and x['tensor_sha256']==source['tensor_sha256'];assert numeric(bank[split][i])==x['numeric_sha256']==source['candidate_numeric_sha256'][state]
   available=True if state=='last' else source['coverage'][state+'_available'];assert x['available']==available;banks_verified+=1
preserved=read(r/'v1_preserved_inventory.json')
for n,v in preserved['files'].items():assert sha(r/n)==v['sha256'] and (r/n).stat().st_size==v['bytes']
v1=[json.loads(x) for x in (r/'generation/records.jsonl').read_text().splitlines()]
for x in v1:
 old=read(r/x['trace_file']);new=read(r/byrecord[x['group'],x['key']]['trace_file']);assert [q['generated_ids'] for q in old['rounds']]==[q['generated_ids'] for q in new['rounds']] and old['state_tensor_sha256']==new['state_tensor_sha256']
out={'ok':True,'main_generation_records':len(records),'candidate_state_tensors_rebuilt':states_verified,'actual_decode_steps_verified':step_values,'body_input_positions_verified':body_positions,'state_bank_rows_verified':banks_verified,'baseline_last_states_and_tokens_exact':273,'warm_anchor_matrices_verified':2,'v1_records_preserved_and_replayed':len(v1),'dangling_tool_tags_in_claimed_strict_answers':dangling,'prompt_selection_train_only_rebuilt':True,'Agent_optimizer_updates':result['Agent_optimizer_updates'],'readout_optimizer_updates':0,'protocol_gate_passed':selection['protocol_gate_passed']}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
