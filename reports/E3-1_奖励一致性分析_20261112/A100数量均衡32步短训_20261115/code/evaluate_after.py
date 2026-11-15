from pathlib import Path
import time
import json
from milestone_common import ROOT,PLAN,setup,seed,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout_eos import rollout,tensor_sha
from stage_io import read,dump,sha
torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
training=read(ROOT/'training/result.json');assert training['ok'] and training['optimizer_updates']==32
checkpoint=ROOT/training['checkpoint'];model,processor=load_policy(checkpoint,False);restore_adapter_exact(model,checkpoint)
expected=training['final_parameter_hash'];assert lora_hash(model)==expected and all(not p.requires_grad for p in model.parameters())
cases=read(ROOT/'inputs/evaluation_cases.json');order={x['image']:i for i,x in enumerate(read(ROOT/'inputs/cases.json'))};images=case_images(cases);tools=real_tools();out=ROOT/'after';out.mkdir();records=[]
for item in cases:
    for role,ss,temp in [('greedy',0,0.)]+[('candidate',800000+order[item['image']]*4+k,1.) for k in range(4)]:
        seed(ss);h,tr=rollout(model,processor,images[item['image']]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp)
        stem=f"after__{item['split']}_{item['row']}_{role}_{ss}";sf=out/'priors'/(stem+'.pt');sf.parent.mkdir(exist_ok=True);torch.save(h,sf);tf=out/'traces'/(stem+'.json');dump(tf,tr)
        x={'id':stem,'arm':'after','image':item['image'],'split':item['split'],'row':item['row'],'role':role,'seed':ss,'temperature':temp,
          'quality_usable':tr['quality_usable'],'strict_protocol_usable':tr['strict_protocol_usable'],'tools_used':tr['tools_used'],'stop_reason':tr['stop_reason'],
          'state_file':str(sf.relative_to(ROOT)),'state_file_sha256':sha(sf),'generation_state_sha256':tensor_sha(h),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf)}
        records.append(x)
        with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x)+'\n')
        dump(ROOT/'progress.json',{'phase':'after_generation','completed':len(records),'total':425});print('A100_AFTER_GENERATED',len(records),'/425',flush=True)
assert len(records)==425 and lora_hash(model)==expected
dump(out/'result.json',{'ok':True,'records':425,'policy_hash_unchanged':expected,'optimizer_updates_in_evaluation':0,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
