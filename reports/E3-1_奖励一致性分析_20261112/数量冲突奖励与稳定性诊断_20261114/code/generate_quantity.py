import argparse
import json
from pathlib import Path
import time

from milestone_common import ROOT,PLAN,setup,seed,dump,sha,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout import rollout,tensor_sha
torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','all'],required=True);a=p.parse_args()
all_cases=json.loads((ROOT/'inputs/cases.json').read_text());cases=all_cases[:1] if a.mode=='smoke' else all_cases
out=ROOT/('smoke' if a.mode=='smoke' else 'generation');out.mkdir(exist_ok=False)
checkpoint=Path(PLAN['policy']['checkpoint']);assert sha(checkpoint/'adapter_model.safetensors')==PLAN['policy']['adapter_sha256']
model,processor=load_policy(checkpoint,False);restored=restore_adapter_exact(model,checkpoint)
expected=PLAN['policy']['lora_tensor_hash'];assert lora_hash(model)==expected and all(not x.requires_grad for x in model.parameters())
dump(out/'settings.json',{'policy_lora_tensor_hash':expected,'restoration':restored,'policy_frozen':True,'optimization_updates':0,'prompt':PLAN['prompt']})
images=case_images(cases);tools=real_tools();records=[]
for i,item in enumerate(cases):
    settings=[('greedy',0,0.)]+[('candidate',800000+i*4+g,1.) for g in range(2 if a.mode=='smoke' else 4)]
    for role,sample_seed,temp in settings:
        seed(sample_seed);image=images[item['image']]['image']
        hidden,trace=rollout(model,processor,image,PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp)
        stem=f"{item['split']}_{item['row']}_{role}_{sample_seed}";path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True);torch.save(hidden,path)
        loaded=torch.load(path,map_location='cpu',weights_only=True);assert torch.equal(loaded,hidden)
        dump(out/'traces'/(stem+'.json'),trace)
        x={'id':a.mode+'__'+stem,'image':item['image'],'split':item['split'],'row':item['row'],'image_order':i,'role':role,'seed':sample_seed,
           'temperature':temp,'quality_usable':trace['quality_usable'],'tools_used':trace['tools_used'],'stop_reason':trace['stop_reason'],
           'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),'generation_state_sha256':tensor_sha(hidden),
           'cache_state_sha256':tensor_sha(loaded),'trace_file':str((out/'traces'/(stem+'.json')).relative_to(ROOT)),
           'trace_sha256':sha(out/'traces'/(stem+'.json'))}
        records.append(x)
        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
        dump(ROOT/'progress.json',{'phase':'generation','mode':a.mode,'completed':len(records),'total':3 if a.mode=='smoke' else 500})
        print('GENERATED',a.mode,len(records),'/',3 if a.mode=='smoke' else 500,trace['stop_reason'],flush=True)
    if a.mode=='smoke':
        _,witness=rollout(model,processor,image,PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],force_initial_tool=True)
        assert witness['tools_used']>=1;dump(out/'actual_tool_software_witness.json',witness)
assert lora_hash(model)==expected and len(records)==(3 if a.mode=='smoke' else 500)
dump(out/'result.json',{'ok':True,'records':len(records),'quality_pass':sum(x['quality_usable'] for x in records),
    'real_tool_executions':sum(x['tools_used'] for x in records),'lora_tensor_hash_unchanged':expected,
    'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
