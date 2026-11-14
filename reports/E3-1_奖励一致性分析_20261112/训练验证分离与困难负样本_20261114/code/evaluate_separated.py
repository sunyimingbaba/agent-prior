import argparse
import json
import time

from milestone_common import ROOT,PLAN,setup,seed,dump,sha,table_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout import rollout,tensor_sha
torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
p=argparse.ArgumentParser();p.add_argument('--policy',required=True);a=p.parse_args()
branch=json.loads((ROOT/'pilot/branch_decision.json').read_text());assert a.policy in branch['policies']
out=ROOT/'evaluation'/a.policy;out.mkdir(parents=True,exist_ok=False)
checkpoint=None if a.policy=='baseline' else ROOT/'training'/a.policy/'checkpoint'
expected=PLAN['inherited_initial_hash'] if checkpoint is None else json.loads((checkpoint.parent/'result.json').read_text())['final_lora_tensor_hash']
model,processor=load_policy(checkpoint,False);restoration={'initialization':True}
if checkpoint is not None:restoration=restore_adapter_exact(model,checkpoint)
assert lora_hash(model)==expected
dump(out/'settings.json',{'policy':a.policy,'split':'val','lora_tensor_hash':expected,'restoration':restoration})
images=table_images(PLAN['validation_rows'],'val');tools=real_tools();records=[]
for row in PLAN['validation_rows']:
    for role,sample_seed,temp in [('greedy',0,0.)]+[('candidate',s,1.) for s in PLAN['evaluation']['sample_seeds']]:
        seed(sample_seed);item=images[row]
        hidden,trace=rollout(model,processor,item['image'],item['caption'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp)
        stem=f'{row}_{role}_{sample_seed}';path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True);torch.save(hidden,path)
        restored=torch.load(path,map_location='cpu',weights_only=True);assert torch.equal(restored,hidden)
        dump(out/'traces'/(stem+'.json'),trace)
        x={'id':a.policy+'__'+stem,'policy':a.policy,'row':row,'image':item['name'],'split':'val','role':role,'seed':sample_seed,
           'temperature':temp,'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),
           'generation_state_sha256':tensor_sha(hidden),'cache_state_sha256':tensor_sha(restored),
           'trace_file':str((out/'traces'/(stem+'.json')).relative_to(ROOT)),'trace_sha256':sha(out/'traces'/(stem+'.json')),
           'quality_usable':trace['quality_usable'],'tools_used':trace['tools_used'],'stop_reason':trace['stop_reason']}
        records.append(x)
        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
        dump(ROOT/'progress.json',{'phase':'evaluation','policy':a.policy,'completed':len(records),'total':150})
        print('EVAL',a.policy,len(records),'/150',trace['stop_reason'],flush=True)
assert len(records)==150 and lora_hash(model)==expected
dump(out/'result.json',{'ok':True,'policy':a.policy,'records':150,'quality_pass':sum(x['quality_usable'] for x in records),
     'lora_tensor_hash':expected,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
