import argparse
import json
from pathlib import Path
import time
from milestone_common import ROOT,PLAN,setup,seed,dump,sha,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout_protocol import rollout,STRICT_TOOL,STRICT_ANSWER
from live_rollout_legacy import rollout as legacy_rollout,tensor_sha

p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['c742_strict','sft_strict','sft_legacy_greedy']);a=p.parse_args()
torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
cases=json.loads((ROOT/'inputs/evaluation_cases.json').read_text());all_cases=json.loads((ROOT/'inputs/cases.json').read_text())
order={x['image']:i for i,x in enumerate(all_cases)};arm=next(x for x in PLAN['evaluation']['arms'] if x['name']==a.arm)
out=ROOT/'evaluation'/a.arm;out.mkdir(parents=True,exist_ok=False)
checkpoint=Path(PLAN['policy']['checkpoint']) if arm['policy']=='inherited' else ROOT/'training/final/checkpoint'
model,processor=load_policy(checkpoint,False);restore=restore_adapter_exact(model,checkpoint)
expected=PLAN['policy']['lora_tensor_hash'] if arm['policy']=='inherited' else json.loads((ROOT/'training/result.json').read_text())['final_parameter_hash']
assert lora_hash(model)==expected and all(not p.requires_grad for p in model.parameters())
dump(out/'settings.json',{'arm':arm,'policy_frozen':True,'policy_lora_tensor_hash':expected,'optimization_updates_during_evaluation':0,
 'prompt_mode':'legacy','restoration':restore})
images=case_images(cases);tools=real_tools();records=[];total=85*(1+arm['samples_per_image'])
for item in cases:
    schedule=[('greedy',0,0.)]+[('candidate',800000+order[item['image']]*4+k,1.) for k in range(arm['samples_per_image'])]
    for role,sample_seed,temp in schedule:
        seed(sample_seed);args=(model,processor,images[item['image']]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'])
        hidden,trace=legacy_rollout(*args,temperature=temp) if arm['controller']=='legacy' else rollout(*args,temperature=temp,prompt_mode='legacy')
        answer=STRICT_ANSWER.fullmatch(trace['rounds'][-1]['generated_text'].strip())
        strict=bool(trace['quality_usable'] and answer and answer.group(1).strip() and all(STRICT_TOOL.fullmatch(z['generated_text'].strip()) for z in trace['rounds'][:-1]))
        stem=f"{a.arm}__{item['split']}_{item['row']}_{role}_{sample_seed}"
        path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True);torch.save(hidden,path)
        loaded=torch.load(path,map_location='cpu',weights_only=True);assert torch.equal(loaded,hidden)
        trace_path=out/'traces'/(stem+'.json');dump(trace_path,trace)
        x={'id':stem,'arm':a.arm,'controller':arm['controller'],'image':item['image'],'split':item['split'],'row':item['row'],
          'image_order':order[item['image']],'role':role,'seed':sample_seed,'temperature':temp,
          'quality_usable':trace['quality_usable'],'strict_protocol_usable':strict,'tools_used':trace['tools_used'],'stop_reason':trace['stop_reason'],
          'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),'generation_state_sha256':tensor_sha(hidden),
          'cache_state_sha256':tensor_sha(loaded),'trace_file':str(trace_path.relative_to(ROOT)),'trace_sha256':sha(trace_path)}
        records.append(x)
        with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x)+'\n')
        dump(ROOT/'progress.json',{'phase':'evaluation','arm':a.arm,'completed':len(records),'total':total})
        print('EVALUATED',a.arm,len(records),'/',total,trace['stop_reason'],flush=True)
assert len(records)==total and lora_hash(model)==expected
dump(out/'result.json',{'ok':True,'arm':a.arm,'records':total,'lora_tensor_hash_unchanged':expected,
 'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
