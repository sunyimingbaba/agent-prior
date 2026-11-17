"""Versioned resume: preserve the651 frozen records, replay3 anchors, finish21 val images."""
import datetime
import io
import json
import os
import time
from pathlib import Path
import pyarrow as pa
from PIL import Image
from milestone_common import ROOT,PLAN,ASSETS,setup,seed,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from prompt_design import messages
from protocol_rollout_v2 import rollout,tensor_sha
from stage_io import read,dump,sha

torch=setup();d=read(ROOT/'protocol_plan.json');started=time.monotonic()
for n,h in read(ROOT/'generation_code_hashes_v2.json').items():assert sha(ROOT/n)==h,n
for n,h in read(ROOT/'resume_code_hashes_20261117.json').items():assert sha(ROOT/n)==h,n
resume=ROOT/'resume_20261117';assert read(resume/'preflight.json')['ok'];before_elapsed=read(resume/'before/generation_progress_v2.json')['elapsed_seconds']
assert read(ROOT/'inputs/prepared.json')['ok'];cohort=read(ROOT/'inputs/cohort.json');train=[c for c in cohort if c['split']=='train'];val=[c for c in cohort if c['split']=='val']
tables={s:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
def image(c):return Image.open(io.BytesIO(tables[c['split']]['image'][c['row']].as_py())).convert('RGB')
seed(0);model,processor=load_policy(Path(d['policy']['checkpoint']),False);restore_adapter_exact(model,Path(d['policy']['checkpoint']));current_policy_hash=lora_hash(model);assert current_policy_hash==d['policy']['lora_hash']
from agent.rollout import SYSTEM_PROMPT
out=ROOT/'generation_v2';out.mkdir(exist_ok=True);ledger=out/'records.jsonl';done={}
if ledger.exists():
 for line in ledger.read_text().splitlines():
  x=json.loads(line);assert sha(ROOT/x['tensor_file'])==x['tensor_sha256'] and sha(ROOT/x['trace_file'])==x['trace_sha256'];done[x['group'],x['key']]=x
def generate(group,prompt,c,baseline=False):
 k=(group,c['key'])
 if k in done:
  assert done[k]['policy_hash']==current_policy_hash
  return done[k]
 seed(0);m=messages(prompt,c,SYSTEM_PROMPT);h,steps,tr=rollout(model,processor,image(c),m,{**c['feedback'],'source_image':c['image'],'source_original_trace_sha256':c['feedback_original_trace_sha256']},d['rollout'])
 if baseline:
  assert tensor_sha(h['last'])==c['numeric_state_sha256'],c['image']
  old=read(Path(PLAN['matched_parent_root'])/c['trace_file']);assert [q['generated_ids'] for q in tr['rounds']]==[q['generated_ids'] for q in old['rounds']],c['image']
 folder=out/group;folder.mkdir(parents=True,exist_ok=True);sf=folder/(c['key']+'.pt');tf=folder/(c['key']+'.json')
 assert not sf.exists() and not tf.exists()
 torch.save({'candidates':h,'step_states':steps},sf);dump(tf,tr)
 rec={'group':group,'prompt':prompt,'key':c['key'],'image':c['image'],'split':c['split'],'row':c['row'],'tensor_file':str(sf.relative_to(ROOT)),'tensor_sha256':sha(sf),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf),'candidate_numeric_sha256':{n:tensor_sha(v) for n,v in h.items()},'quality_usable':tr['quality_usable'],'strict_protocol_usable':tr['strict_protocol_usable'],'stop_reason':tr['stop_reason'],'generated_result_tag':tr['generated_result_tag'],'tools_used':tr['tools_used'],'feedback_consumed':any(e['after_round']<len(tr['rounds']) for e in tr['tool_events']),'coverage':tr['state_coverage'],'policy_hash':current_policy_hash}
 with ledger.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 done[k]=rec;dump(ROOT/'generation_progress_resume_20261117.json',{'group':group,'prompt':prompt,'completed_unique_generations':len(done),'image':c['image'],'elapsed_seconds':time.monotonic()-started});
 if len(done)%10==0:print('PROTOCOL_STATE_GENERATION',len(done),group,c['image'],flush=True)
 return rec
lock=read(ROOT/'prompt_selection_lock_v2.json');chosen=lock['chosen_prompt'];group=lock['selected_train_group'];success=lock['protocol_gate_passed'];fallback=None
assert chosen=='contract_only' and group=='pilot_initial/contract_only' and success and lock['Agent_optimizer_updates']==0 and lock['policy_hash']==current_policy_hash
assert len(done)==651 and all(x['policy_hash']==current_policy_hash for x in done.values())
for name in d['prompt_candidates_priority']:
 xs=[done['pilot_initial/'+name,c['key']] for c in train];valid=sum(x['strict_protocol_usable'] and x['tools_used']>0 and x['feedback_consumed'] for x in xs);bad=sum(x['generated_result_tag'] for x in xs);entry=lock['train_only_outcomes'][name]
 assert entry['strict_completed_with_consumed_tool']==valid and entry['model_generated_result']==bad and entry['passed']==(valid/63>=.95 and bad/63<=.02)
assert all(('baseline',c['key']) in done for c in cohort)
assert [c['key'] for c in val if ('selected_validation',c['key']) in done]==[c['key'] for c in val[:189]]
anchors=[('baseline','legacy',train[0]),('baseline','legacy',train[1]),('selected_validation',chosen,val[188])]
anchor_rows=[]
for source,prompt,c in anchors:
 seed(0);h,steps,tr=rollout(model,processor,image(c),messages(prompt,c,SYSTEM_PROMPT),{**c['feedback'],'source_image':c['image'],'source_original_trace_sha256':c['feedback_original_trace_sha256']},d['rollout'])
 old=done[source,c['key']];saved=torch.load(ROOT/old['tensor_file'],map_location='cpu',weights_only=True);oldtrace=read(ROOT/old['trace_file'])
 assert [x['generated_ids'] for x in tr['rounds']]==[x['generated_ids'] for x in oldtrace['rounds']]
 assert len(steps)==len(saved['step_states']) and all(torch.equal(x,y) for x,y in zip(steps,saved['step_states'])) and all(torch.equal(v,saved['candidates'][k]) for k,v in h.items())
 name=source+'_'+c['key'];sf=resume/(name+'.pt');tf=resume/(name+'.json');assert not sf.exists() and not tf.exists();torch.save({'candidates':h,'step_states':steps},sf);dump(tf,tr)
 anchor_rows.append({'record_group':source,'key':c['key'],'all_steps_and_candidates_exact':True,'tensor_file':str(sf.relative_to(ROOT)),'tensor_sha256':sha(sf),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf)})
dump(resume/'warm_replay.json',{'ok':True,'replays':anchor_rows,'all_original_anchors_preserved':True})
if fallback is None:
 for c in cohort:generate('baseline','legacy',c,True)
for c in val:generate('selected_validation',chosen,c)
assert lora_hash(model)==lock['policy_hash']
bank={};records=[]
for prefix in ['legacy','selected']:
 for state in ['last','answer_prefill','answer_pool']:
  family=prefix+'_'+state;bank[family]={};meta={}
  for split,cs in [('train',train),('val',val)]:
   hs=[];refs=[]
   for c in cs:
    source='baseline' if prefix=='legacy' else (group if split=='train' else 'selected_validation');x=done[source,c['key']];saved=torch.load(ROOT/x['tensor_file'],map_location='cpu',weights_only=True);hs.append(saved['candidates'][state]);refs.append({'key':c['key'],'image':c['image'],'row':c['row'],'record_group':source,'tensor_file':x['tensor_file'],'tensor_sha256':x['tensor_sha256'],'candidate':state,'numeric_sha256':x['candidate_numeric_sha256'][state],'available':True if state=='last' else x['coverage'][state+'_available']})
   bank[family][split]=torch.stack(hs);meta[split]=refs
  folder=ROOT/'state_banks';folder.mkdir(exist_ok=True);torch.save(bank[family],folder/(family+'.pt'));dump(folder/(family+'.json'),meta)
dump(out/'records.json',list(done.values()));dump(ROOT/'generation_result_v2.json',{'ok':True,'protocol_repair_passed':success,'exploratory_after_failed_repair':not success,'cohort_images':273,'prompt_selection':lock,'baseline_exact_last_state_replays':273,'warm_anchors':2,'unique_generations':len(done),'state_families':6,'candidate_states':6*273,'Agent_optimizer_updates':30 if fallback else 0,'readout_optimizer_updates':0,'fallback_training':fallback,'initial_policy_hash':d['policy']['lora_hash'],'final_policy_hash':lora_hash(model),'elapsed_seconds':before_elapsed+time.monotonic()-started,'resume_elapsed_seconds':time.monotonic()-started,'time_accounting':'sum of pre-pause progress duration and resume active duration','resumed_existing_records':651,'new_generations_after_resume':21,'supplemental_exact_warm_replays':3})
print('PROTOCOL_AND_STATE_GENERATION_COMPLETE',chosen,success,flush=True)
