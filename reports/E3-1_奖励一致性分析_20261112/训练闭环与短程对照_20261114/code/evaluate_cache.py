"""All policies use the shared live-tool rollout and the same cache contract."""
import argparse
import json
import time

from milestone_common import ROOT,PLAN,dump,sha,setup,seed,table_images,load_policy,real_tools,lora_hash
from live_rollout import rollout,tensor_sha
from checkpoint_io import restore_adapter_exact
torch=setup()
from agent.rollout import SYSTEM_PROMPT

p=argparse.ArgumentParser();p.add_argument('--policy',choices=['baseline','retrieval','format_only'],required=True)
a=p.parse_args();out=ROOT/'evaluation'/a.policy;out.mkdir(parents=True,exist_ok=False)
started=time.monotonic();seed(0)
checkpoint=None if a.policy=='baseline' else ROOT/'training'/a.policy/'checkpoint'
model,processor=load_policy(checkpoint,trainable=False)
restoration={'original_initialization':True}
if checkpoint is not None:restoration=restore_adapter_exact(model,checkpoint)
policy_hash=lora_hash(model)
if a.policy=='baseline':
    expected=json.loads((ROOT/'training/smoke/result.json').read_text())['initial_lora_tensor_hash']
else:
    expected=json.loads((ROOT/'training'/a.policy/'result.json').read_text())['final_lora_tensor_hash']
assert policy_hash==expected,'Checkpoint reload changed adapter values.'
assert all(not p.requires_grad for p in model.parameters())
tools=real_tools();images=table_images(PLAN['validation_rows']);records=[]
dump(out/'settings.json',{'policy':a.policy,'lora_tensor_hash':policy_hash,'expected_tensor_hash':expected,
                         'checkpoint_restoration':restoration,'rollout_config':PLAN['rollout']})
for row in PLAN['validation_rows']:
    item=images[row]
    for role,sample_seed,temp in [('greedy',0,0.)]+[('candidate',s,1.) for s in PLAN['evaluation']['sample_seeds']]:
        seed(sample_seed)
        hidden,trace=rollout(model,processor,item['image'],item['caption'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp)
        stem=f'{row}_{role}_{sample_seed}'
        path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True)
        torch.save(hidden,path);restored=torch.load(path,map_location='cpu',weights_only=True)
        assert torch.equal(hidden,restored) and tensor_sha(restored)==trace['state_tensor_sha256']
        dump(out/'traces'/(stem+'.json'),trace)
        record={'id':a.policy+'__'+stem,'policy':a.policy,'row':row,'image':item['name'],'role':role,
            'seed':sample_seed,'temperature':temp,'state_file':str(path.relative_to(ROOT)),
            'state_file_sha256':sha(path),'generation_state_sha256':trace['state_tensor_sha256'],
            'cache_state_sha256':tensor_sha(restored),'trace_file':str((out/'traces'/(stem+'.json')).relative_to(ROOT)),
            'trace_sha256':sha(out/'traces'/(stem+'.json')),'quality_usable':trace['quality_usable'],
            'stop_reason':trace['stop_reason'],'tools_used':trace['tools_used'],
            'rounds':len(trace['rounds']),'generated_tokens':sum(x['generated_token_count'] for x in trace['rounds'])}
        records.append(record)
        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        dump(ROOT/'progress.json',{'phase':'evaluation_cache','policy':a.policy,'completed':len(records),'total':110})
        print('EVAL',a.policy,len(records),'/110',stem,trace['stop_reason'],'tools',trace['tools_used'],flush=True)
assert len(records)==PLAN['evaluation']['expected_trajectories_per_policy']
assert lora_hash(model)==policy_hash
result={'ok':True,'policy':a.policy,'records':len(records),'all_generation_cache_hashes_equal':True,
    'policy_tensor_hash_before':policy_hash,'policy_tensor_hash_after':policy_hash,
    'quality_pass':sum(x['quality_usable'] for x in records),'actual_tool_executions':sum(x['tools_used'] for x in records),
    'elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
dump(out/'result.json',result);print(json.dumps(result),flush=True)
