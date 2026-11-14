import argparse
import json
from pathlib import Path
import shutil
import time

from milestone_common import ROOT,PLAN,dump,sha,setup,seed,table_images,load_policy,real_tools,lora_hash
from live_rollout import rollout,tensor_sha
from checkpoint_io import restore_adapter_exact
torch=setup()
from agent.rollout import SYSTEM_PROMPT

p=argparse.ArgumentParser();p.add_argument('--policy',choices=['baseline','retrieval','format_only','aligned_margin'],required=True)
a=p.parse_args();out=ROOT/'evaluation'/a.policy;out.mkdir(parents=True,exist_ok=False)
started=time.monotonic();seed(0)
previous=Path(PLAN['previous_stage_root'])
if a.policy=='baseline':checkpoint=None;expected=PLAN['inherited_initial_hash']
elif a.policy=='aligned_margin':
    checkpoint=ROOT/'training/aligned_margin/checkpoint'
    expected=json.loads((ROOT/'training/aligned_margin/result.json').read_text())['final_lora_tensor_hash']
else:
    source=PLAN['inherited_policies'][a.policy];checkpoint=Path(source['checkpoint_path']);expected=source['lora_tensor_hash']
    assert sha(checkpoint/'adapter_model.safetensors')==source['adapter_file_sha256']
model,processor=load_policy(checkpoint,trainable=False)
restoration={'original_initialization':True}
if checkpoint is not None:restoration=restore_adapter_exact(model,checkpoint)
assert lora_hash(model)==expected
tools=real_tools();records=[];imported=0
if a.policy!='aligned_margin':
    for line in (previous/'evaluation'/a.policy/'records.jsonl').read_text().splitlines():
        x=json.loads(line)
        if x['role']=='candidate' and x['seed'] not in PLAN['evaluation']['sample_seeds']:continue
        assert x['row'] in PLAN['validation_groups']['legacy22']
        for field,hash_field in [('state_file','state_file_sha256'),('trace_file','trace_sha256')]:
            source=previous/x[field];target=ROOT/x[field]
            assert sha(source)==x[hash_field];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        records.append({**x,'reused_previous':True,'origin_stage':str(previous)})
        imported+=1
    assert imported==66
rows=PLAN['validation_groups']['new50'] if imported else PLAN['validation_rows']
images=table_images(rows)
dump(out/'settings.json',{'policy':a.policy,'lora_tensor_hash':expected,'checkpoint_restoration':restoration,
                         'generated_rows':rows,'imported_legacy_trajectories':imported})
with (out/'records.jsonl').open('w') as f:
    for x in records:f.write(json.dumps(x,ensure_ascii=False)+'\n')
for row in rows:
    item=images[row]
    for role,sample_seed,temp in [('greedy',0,0.)]+[('candidate',s,1.) for s in PLAN['evaluation']['sample_seeds']]:
        seed(sample_seed)
        hidden,trace=rollout(model,processor,item['image'],item['caption'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp)
        stem=f'{row}_{role}_{sample_seed}';path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True)
        torch.save(hidden,path);restored=torch.load(path,map_location='cpu',weights_only=True)
        assert torch.equal(hidden,restored) and tensor_sha(restored)==trace['state_tensor_sha256']
        dump(out/'traces'/(stem+'.json'),trace)
        record={'id':a.policy+'__'+stem,'policy':a.policy,'row':row,'image':item['name'],'role':role,'seed':sample_seed,
            'temperature':temp,'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),
            'generation_state_sha256':trace['state_tensor_sha256'],'cache_state_sha256':tensor_sha(restored),
            'trace_file':str((out/'traces'/(stem+'.json')).relative_to(ROOT)),'trace_sha256':sha(out/'traces'/(stem+'.json')),
            'quality_usable':trace['quality_usable'],'stop_reason':trace['stop_reason'],'tools_used':trace['tools_used'],
            'rounds':len(trace['rounds']),'generated_tokens':sum(x['generated_token_count'] for x in trace['rounds']),
            'reused_previous':False,'origin_stage':str(ROOT)}
        records.append(record)
        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        dump(ROOT/'progress.json',{'phase':'evaluation','policy':a.policy,'completed':len(records),'total':216,'imported':imported})
        print('EVAL',a.policy,len(records),'/216',stem,trace['stop_reason'],flush=True)
assert len(records)==216 and lora_hash(model)==expected
dump(out/'result.json',{'ok':True,'policy':a.policy,'records':216,'new_generations':216-imported,
    'imported':imported,'quality_pass':sum(x['quality_usable'] for x in records),'lora_tensor_hash':expected,
    'elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
