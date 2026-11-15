import argparse
from pathlib import Path
import json
import time
from milestone_common import ROOT,PLAN,setup,seed,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout_eos import rollout,tensor_sha
from stage_io import read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--policy-index',type=int,choices=[0,1],required=True);k=p.parse_args().policy_index
torch=setup();seed(0);start=time.monotonic();policy=PLAN['policies'][k];assert read(ROOT/'preflight.json')['ok']
from agent.rollout import SYSTEM_PROMPT
model,processor=load_policy(Path(policy['checkpoint']),False);restored=restore_adapter_exact(model,Path(policy['checkpoint']));assert lora_hash(model)==policy['lora_hash'] and all(not p.requires_grad for p in model.parameters())
cases=read(ROOT/'inputs/cases.json');images=case_images(cases);tools=real_tools();out=ROOT/'generation'/policy['name'];out.mkdir(parents=True);old=next(x for x in read(ROOT/'historical/records.json') if x['policy']==policy['name'] and x['image']=='102.tif' and x['role']=='greedy')
seed(0);h,tr=rollout(model,processor,images['102.tif']['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=0.)
anchor=ROOT/'anchors'/policy['name'];anchor.mkdir(parents=True);torch.save(h,anchor/'prior.pt');dump(anchor/'trace.json',tr);trace=read(ROOT/old['trace_file']);assert tensor_sha(h)==old['generation_state_sha256'] and [z['generated_ids'] for z in tr['rounds']]==[z['generated_ids'] for z in trace['rounds']]
dump(anchor/'result.json',{'ok':True,'policy_hash':policy['lora_hash'],'generation_state_sha256':tensor_sha(h),'parent_greedy_id':old['id'],'exact_tokens_and_state_equal':True,'probe_generations':1,'optimizer_updates':0})
records=[]
for image_index,item in enumerate(cases):
    for sample in range(16):
        ss=2100000+k*10000+image_index*16+sample;seed(ss);h,tr=rollout(model,processor,images[item['image']]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=1.)
        stem=f"{policy['name']}__{item['image']}__s{sample:02d}__{ss}";sf=out/'priors'/(stem+'.pt');sf.parent.mkdir(exist_ok=True);torch.save(h,sf);tf=out/'traces'/(stem+'.json');dump(tf,tr)
        row={'id':stem,'policy':policy['name'],'policy_index':k,'image':item['image'],'image_index':image_index,'actual_count':item['actual_count'],'sample':sample,'seed':ss,'role':'candidate','quality_usable':tr['quality_usable'],'strict_protocol_usable':tr['strict_protocol_usable'],'tools_used':tr['tools_used'],'stop_reason':tr['stop_reason'],'state_file':str(sf.relative_to(ROOT)),'state_file_sha256':sha(sf),'generation_state_sha256':tensor_sha(h),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf),'policy_hash':policy['lora_hash']}
        records.append(row)
        with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        dump(ROOT/'progress.json',{'phase':'generation','policy':policy['name'],'completed_policy':len(records),'completed_total':k*128+len(records),'total':256});print('PERMUTATION_GENERATED',k*128+len(records),'/256',flush=True)
assert len(records)==128 and lora_hash(model)==policy['lora_hash']
dump(out/'result.json',{'ok':True,'records':128,'policy_hash_unchanged':policy['lora_hash'],'optimizer_updates':0,'warm_anchor_generations':1,'saved_fp32_adapter_exact':restored,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});print('POLICY_GENERATION_COMPLETE',policy['name'],flush=True)
