"""Frozen 2x2x2 generation interventions, with exact baseline anchors and provenance."""
import datetime
import gc
import hashlib
import io
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
import pyarrow as pa
from PIL import Image
from milestone_common import ROOT,PLAN,ASSETS,setup,seed,load_policy,lora_hash
from checkpoint_io import restore_adapter_exact
from counterfactual_rollout import rollout,tensor_sha
from stage_io import read,dump,sha

torch=setup();started=time.monotonic();p=Path(PLAN['matched_parent_root']);d=read(ROOT/'diagnostic_plan.json')
for n,h in read(ROOT/'generation_code_hashes.json').items():assert sha(ROOT/n)==h,n
assert read(ROOT/'inputs/prepared.json')['ok']
cohort=read(ROOT/'inputs/cohort.json');constant=read(ROOT/'inputs/constant_feedback.json')
tables={s:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
def original(c):
 img=Image.open(io.BytesIO(tables[c['split']]['image'][c['row']].as_py())).convert('RGB')
 assert list(img.size)==c['size'] and hashlib.sha256(img.tobytes()).hexdigest()==c['rgb_sha256'];return img
def tool(c,condition):
 source=constant if condition['feedback']=='constant_training102' else c
 def call(image):
  return {**source['feedback'],'feedback_source':condition['feedback'],'source_image':source['image'],'source_split':source['split'],'source_trace_sha256':source['feedback_original_trace_sha256']}
 return {'scene_classify':call}
out=ROOT/'generation';out.mkdir(exist_ok=True);ledger=out/'records.jsonl';done={};actual=0
if ledger.exists():
 for line in ledger.read_text().splitlines():
  c=json.loads(line);k=(c['condition'],c['key']);assert k not in done
  assert sha(ROOT/c['state_file'])==c['state_file_sha256'] and sha(ROOT/c['trace_file'])==c['trace_sha256']
  done[k]=c;actual+=int(c['new_actual_generate'])
base=d['conditions'][0];assert base['baseline_reused']
with ledger.open('a',encoding='utf-8') as f:
 for c in cohort:
  k=(base['name'],c['key'])
  if k in done:continue
  sf=out/'baseline'/('states/'+c['key']+'.pt');tf=out/'baseline'/('traces/'+c['key']+'.json')
  for src,dst,h in [(p/c['state_file'],sf,c['state_file_sha256']),(p/c['trace_file'],tf,c['trace_sha256'])]:
   assert sha(src)==h;dst.parent.mkdir(parents=True,exist_ok=True)
   if dst.exists():assert sha(dst)==h
   else:shutil.copy2(src,dst)
  x={'condition':base['name'],'key':c['key'],'split':c['split'],'row':c['row'],'image':c['image'],'rgb_sha256':c['rgb_sha256'],'state_file':str(sf.relative_to(ROOT)),'trace_file':str(tf.relative_to(ROOT)),'state_file_sha256':sha(sf),'trace_sha256':sha(tf),'numeric_state_sha256':c['numeric_state_sha256'],'new_actual_generate':False,'baseline_reused':True,'canonical_reused':False,'quality_usable':c['quality_usable'],'stop_reason':c['stop_reason'],'tools_used':c['tools_used'],'feedback_source_image':c['image'],'feedback_original_trace_sha256':c['feedback_original_trace_sha256']}
  f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());done[k]=x
seed(0);policy=PLAN['policies'][0];model,processor=load_policy(Path(policy['checkpoint']),False);restore_adapter_exact(model,Path(policy['checkpoint']))
assert lora_hash(model)==d['policy_lora_hash']
from agent.rollout import SYSTEM_PROMPT
for c in cohort[:2]:
 seed(0);h,tr=rollout(model,processor,original(c),d['query'],tool(c,base),SYSTEM_PROMPT,d['rollout'],temperature=0.,prompt_mode=base['example'])
 assert tensor_sha(h)==c['numeric_state_sha256'],c['image']
 old=read(p/c['trace_file']);assert [x['generated_ids'] for x in tr['rounds']]==[x['generated_ids'] for x in old['rounds']]
 dump(out/('anchor_'+c['key']+'.json'),tr)
with ledger.open('a',encoding='utf-8') as f:
 for condition in d['conditions'][1:]:
  canonical={x['signature']:x for x in done.values() if x['condition']==condition['name'] and 'signature' in x}
  for c in cohort:
   k=(condition['name'],c['key'])
   if k in done:continue
   image=original(c)
   if condition['policy_image']=='gray127':image=Image.new('RGB',image.size,(127,127,127))
   source=constant if condition['feedback']=='constant_training102' else c
   signature=hashlib.sha256(json.dumps({'condition':condition['name'],'rgb':hashlib.sha256(image.tobytes()).hexdigest(),'size':image.size,'feedback':source['feedback'],'query':d['query'],'seed':0,'lora':d['policy_lora_hash']},sort_keys=True).encode()).hexdigest()
   reuse=condition['policy_image']=='gray127' and condition['feedback']=='constant_training102' and signature in canonical
   if reuse:
    old=canonical[signature];x={**old,'key':c['key'],'split':c['split'],'row':c['row'],'image':c['image'],'rgb_sha256':c['rgb_sha256'],'new_actual_generate':False,'canonical_reused':True,'canonical_key':old['key']}
   else:
    seed(0);h,tr=rollout(model,processor,image,d['query'],tool(c,condition),SYSTEM_PROMPT,d['rollout'],temperature=0.,prompt_mode=condition['example'])
    assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all()
    sf=out/condition['name']/('states/'+c['key']+'.pt');tf=out/condition['name']/('traces/'+c['key']+'.json');sf.parent.mkdir(parents=True,exist_ok=True);tf.parent.mkdir(parents=True,exist_ok=True)
    for path in [sf,tf]:
     if path.exists():
      orphan=out/'interrupted_uncommitted'/str(time.time_ns())/path.name;orphan.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,orphan)
    torch.save(h,sf);dump(tf,tr);actual+=1;assert actual<=1911
    x={'condition':condition['name'],'key':c['key'],'split':c['split'],'row':c['row'],'image':c['image'],'rgb_sha256':c['rgb_sha256'],'state_file':str(sf.relative_to(ROOT)),'trace_file':str(tf.relative_to(ROOT)),'state_file_sha256':sha(sf),'trace_sha256':sha(tf),'numeric_state_sha256':tensor_sha(h),'new_actual_generate':True,'baseline_reused':False,'canonical_reused':False,'quality_usable':tr['quality_usable'],'stop_reason':tr['stop_reason'],'tools_used':tr['tools_used'],'signature':signature,'feedback_source_image':source['image'],'feedback_original_trace_sha256':source['feedback_original_trace_sha256']}
    canonical[signature]=x
   assert x['feedback_source_image']==source['image']
   f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());done[k]=x
   progress={'condition':condition['name'],'logical_completed':len(done),'logical_total':2184,'new_actual_generations':actual,'latest_image':c['image'],'elapsed_seconds':time.monotonic()-started,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   dump(ROOT/'generation_progress.json',progress)
   if len(done)%10==0:print('DIAGNOSTIC_GENERATION',json.dumps(progress),flush=True)
assert len(done)==2184 and lora_hash(model)==d['policy_lora_hash']
records=sorted(done.values(),key=lambda x:(x['condition'],x['split'],x['row']));dump(out/'records.json',records)
dump(ROOT/'generation_result.json',{'ok':True,'logical_states':len(records),'new_actual_generations':actual,'warm_baseline_replays':2,'baseline_reused':273,'canonical_reused':sum(x['canonical_reused'] for x in records),'quality_by_condition':{c['name']:dict(Counter(x['stop_reason'] for x in records if x['condition']==c['name'])) for c in d['conditions']},'Agent_optimizer_updates':0,'readout_optimizer_updates':0,'lora_hash_unchanged':lora_hash(model),'elapsed_seconds':time.monotonic()-started})
print('ALL_FACTORIAL_GENERATION_COMPLETE',flush=True)
