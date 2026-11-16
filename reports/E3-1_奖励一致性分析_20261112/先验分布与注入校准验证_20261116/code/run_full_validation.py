import io
import json
import time
from pathlib import Path
import pyarrow as pa
from PIL import Image
from milestone_common import ROOT,PLAN,ASSETS,setup,seed,load_policy,real_tools,lora_hash,case_images
from checkpoint_io import restore_adapter_exact
from live_rollout_eos import rollout,tensor_sha
from stage_io import read,dump,sha
import hashlib

torch=setup();seed(0);out=ROOT/'full_validation';out.mkdir();fp=read(ROOT/'full_validation_plan.json');assert read(ROOT/'selection.json')['selected_variant']=='mean_shift_auto'
path=ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow';table=pa.ipc.open_file(str(path)).read_all();assert table.num_rows==210;cases=[]
for i in range(210):
 image=Image.open(io.BytesIO(table['image'][i].as_py())).convert('RGB');caps=table['caption'][i].as_py();cases.append({'image':table['path'][i].as_py(),'row':i,'split':'val','image_rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'original_first_caption':caps[0],'captions':caps})
dump(out/'cases.json',cases);old=[x for x in read(ROOT/'states/records.json') if x['family']=='balanced32' and x['role']=='greedy' and x['split']=='val'];assert len(old)==22;oldmap={x['image']:x for x in old};by={x['image']:x for x in cases};assert set(oldmap)<=set(by);images=case_images(cases);p=PLAN['policies'][1];model,processor=load_policy(Path(p['checkpoint']),False);restore_adapter_exact(model,Path(p['checkpoint']));assert lora_hash(model)==p['lora_hash']
from agent.rollout import SYSTEM_PROMPT
tools=real_tools();records=[]
for name in [old[0]['image'],old[1]['image']]:
 seed(0);h,tr=rollout(model,processor,images[name]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=0.);assert tensor_sha(h)==oldmap[name]['numeric_state_sha256'];dump(out/('anchor_'+name+'.json'),tr)
for c in cases:
 n=c['image']
 if n in oldmap:records.append({**oldmap[n],'reused':True});continue
 seed(0);h,tr=rollout(model,processor,images[n]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=0.);sid='fullval__'+n;sf=out/'priors'/(sid+'.pt');sf.parent.mkdir(exist_ok=True);torch.save(h,sf);tf=out/'traces'/(sid+'.json');dump(tf,tr)
 records.append({'id':sid,'family':'balanced32','image':n,'row':c['row'],'split':'val','role':'greedy','reused':False,'state_file':str(sf.relative_to(ROOT)),'state_file_sha256':sha(sf),'numeric_state_sha256':tensor_sha(h),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf),'quality_usable':tr['quality_usable'],'strict_protocol_usable':tr['strict_protocol_usable'],'tools_used':tr['tools_used'],'stop_reason':tr['stop_reason']})
 dump(ROOT/'full_validation_progress.json',{'phase':'generation','new_completed':sum(not x['reused'] for x in records),'new_total':188});print('FULL_VAL_NEW_GENERATED',sum(not x['reused'] for x in records),'/188',flush=True)
assert len(records)==210 and sum(not x['reused'] for x in records)==188 and lora_hash(model)==p['lora_hash'];dump(out/'records.json',records);dump(out/'generation_result.json',{'ok':True,'new_states':188,'reused_states':22,'warm_probes':2,'policy_hash_unchanged':p['lora_hash'],'optimizer_updates':0,'arrow_sha256':sha(path)});print('FULL_VAL_GENERATION_COMPLETE',flush=True)
