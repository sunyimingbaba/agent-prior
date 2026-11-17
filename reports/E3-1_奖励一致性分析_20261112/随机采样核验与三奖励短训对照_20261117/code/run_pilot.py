"""Train8 stochastic protocol choice and same-image reward information, no policy updates."""
import datetime
from pathlib import Path
import time
import numpy as np
from milestone_common import ROOT,setup,seed,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from stochastic_rollout import rollout
from prompt_design import messages
from reward_model import RewardScorer
from stage_io import read,dump,sha
torch=setup();p=read(ROOT/'plan.json');assert read(ROOT/'prepare_result.json')['ok'];assert not (ROOT/'pilot/result.json').exists();started=time.monotonic()
seed(0);model,processor=load_policy(Path(p['policy']['checkpoint']),False);restore_adapter_exact(model,Path(p['policy']['checkpoint']));assert lora_hash(model)==p['policy']['lora_hash'];scorer=RewardScorer();by={x['image']:x for x in read(ROOT/'inputs/cohort.json')}
from agent.rollout import SYSTEM_PROMPT
out=ROOT/'pilot';out.mkdir();records=[];outcomes={};selected=None
for prompt in p['prompt_priority']:
 rows=[]
 for i,name in enumerate(p['training_images']):
  c=by[name]
  for sample in range(8):
   ss=2026111700+100*i+sample;seed(ss);h,tensors,tr=rollout(model,processor,scorer.case_image(c),messages(prompt,c,SYSTEM_PROMPT),{**c['feedback'],'source_image':name,'source_original_trace_sha256':c['feedback_original_trace_sha256']},p['rollout'],temperature=1.,capture=(i==sample==0))
   values=scorer.score(h,c,tr['strict_completed_consumed_tool']);folder=out/prompt;folder.mkdir(exist_ok=True);stem=c['key']+f'_{sample:02d}';sf=folder/(stem+'.pt');tf=folder/(stem+'.json');torch.save({**tensors,**scorer.last_tensors},sf);dump(tf,tr)
   row={'prompt':prompt,'image':name,'key':c['key'],'split':'train','row':c['row'],'sample':sample,'seed':ss,'strict_completed_consumed_tool':tr['strict_completed_consumed_tool'],'feedback_consumed':tr['feedback_consumed'],'tools_used':tr['tools_used'],'generated_result_tag':tr['generated_result_tag'],'stop_reason':tr['stop_reason'],'policy_hash':p['policy']['lora_hash'],**values,'tensor_file':str(sf.relative_to(ROOT)),'tensor_sha256':sha(sf),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf)};rows.append(row);records.append(row)
   with (out/'records.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n')
   dump(ROOT/'pilot_progress.json',{'prompt':prompt,'completed_records':len(records),'image':name,'sample':sample,'elapsed_seconds':time.monotonic()-started})
  print('STOCHASTIC_PILOT',prompt,i+1,'/8',flush=True)
 valid=sum(x['strict_completed_consumed_tool'] for x in rows);consumed=sum(x['feedback_consumed'] and x['tools_used']>0 for x in rows);bad=sum(x['generated_result_tag'] for x in rows);passed=valid/64>=.90 and consumed/64>=.95 and bad/64<=.02;signal={}
 for arm in p['reward_arms']:
  key='r_'+arm;groups=[]
  for name in p['training_images']:
   rs=np.array([x[key] for x in rows if x['image']==name]);groups.append({'image':name,'reward_std':float(rs.std(ddof=1)),'unique_rewards':len(set(rs.tolist())),'range':float(np.ptp(rs))})
  signal[arm]={'informative_groups':sum(x['reward_std']>1e-8 for x in groups),'groups':groups}
 outcomes[prompt]={'strict_completed':valid,'consumed_tool':consumed,'generated_result':bad,'images':8,'records':64,'passed':passed,'reward_signals':signal}
 if passed:selected=prompt;break
assert lora_hash(model)==p['policy']['lora_hash'];dump(out/'records.json',records);dump(ROOT/'prompt_selection_lock.json',{'selected_prompt':selected,'train_protocol_only':True,'fixed_priority':p['prompt_priority'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policy_hash':p['policy']['lora_hash'],'optimizer_updates':0,'outcomes':outcomes})
dump(out/'result.json',{'ok':True,'selected_prompt':selected,'protocol_gate_passed':selected is not None,'records':len(records),'outcomes':outcomes,'Agent_optimizer_updates':0,'readout_optimizer_updates':0,'readout_immutability':scorer.verify_frozen(),'elapsed_seconds':time.monotonic()-started});print('PILOT_COMPLETE',selected,flush=True)
