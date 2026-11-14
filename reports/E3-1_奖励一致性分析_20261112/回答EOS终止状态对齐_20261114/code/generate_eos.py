import json
from pathlib import Path
import time
from milestone_common import ROOT,PLAN,setup,seed,dump,sha,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout_eos import rollout,tensor_sha,STRICT_TOOL,STRICT_ANSWER

torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
cases=json.loads((ROOT/'inputs/evaluation_cases.json').read_text());order={x['image']:i for i,x in enumerate(json.loads((ROOT/'inputs/cases.json').read_text()))}
source=json.loads((ROOT/'reference_run/scoring/all_scores.json').read_text());old={(x['image'],x['role'],x['seed']):x for x in source if x['arm']=='sft_strict'}
out=ROOT/'generation';out.mkdir(exist_ok=False)
model,processor=load_policy(Path(PLAN['policy']['checkpoint']),False);restored=restore_adapter_exact(model,Path(PLAN['policy']['checkpoint']))
expected=PLAN['policy']['lora_tensor_hash'];assert lora_hash(model)==expected and all(not p.requires_grad for p in model.parameters())
dump(out/'settings.json',{'policy_frozen':True,'optimization_updates':0,'policy_lora_tensor_hash':expected,'restoration':restored,
 'tool_stop':'complete registered tool closing tag','answer_stop':'native configured EOS','prompt_mode':'legacy'})
images=case_images(cases);tools=real_tools();records=[]
for item in cases:
    for role,sample_seed,temp in [('greedy',0,0.)]+[('candidate',800000+order[item['image']]*4+k,1.) for k in range(4)]:
        seed(sample_seed)
        hidden,trace=rollout(model,processor,images[item['image']]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp,prompt_mode='legacy')
        ref=old[item['image'],role,sample_seed];previous=json.loads((ROOT/'reference_run'/ref['trace_file']).read_text())
        prefix=len(trace['rounds'])==len(previous['rounds'])
        if prefix:
            prefix=all(z['prompt_token_ids']==p['prompt_token_ids'] and z['generated_ids'][:len(p['generated_ids'])]==p['generated_ids'] for z,p in zip(trace['rounds'],previous['rounds']))
        same_tools=len(trace['tool_events'])==len(previous['tool_events']) and all({k:v for k,v in x.items() if k!='confidence'}=={k:v for k,v in y.items() if k!='confidence'} and abs(x['confidence']-y['confidence'])<=1e-6 for x,y in zip(trace['tool_events'],previous['tool_events']))
        aa=STRICT_ANSWER.fullmatch(trace['rounds'][-1]['generated_text'].strip())
        strict=bool(trace['quality_usable'] and aa and aa.group(1).strip() and all(STRICT_TOOL.fullmatch(z['generated_text'].strip()) for z in trace['rounds'][:-1]))
        if trace['quality_usable']:assert trace['rounds'][-1]['last_emitted_token_id'] in trace['rounds'][-1]['eos_token_ids']
        stem=f"eos__{item['split']}_{item['row']}_{role}_{sample_seed}"
        path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True);torch.save(hidden,path)
        loaded=torch.load(path,map_location='cpu',weights_only=True);assert torch.equal(hidden,loaded)
        tp=out/'traces'/(stem+'.json');dump(tp,trace)
        record={'id':stem,'arm':'sft_fulltool_eos','image':item['image'],'split':item['split'],'row':item['row'],'image_order':order[item['image']],
          'role':role,'seed':sample_seed,'temperature':temp,'quality_usable':trace['quality_usable'],'strict_protocol_usable':strict,
          'tools_used':trace['tools_used'],'stop_reason':trace['stop_reason'],'reference_id':ref['id'],'reference_prefix_equal':prefix,'reference_tools_equal':same_tools,
          'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),'generation_state_sha256':tensor_sha(hidden),'cache_state_sha256':tensor_sha(loaded),
          'trace_file':str(tp.relative_to(ROOT)),'trace_sha256':sha(tp)}
        records.append(record)
        with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        dump(ROOT/'progress.json',{'phase':'generation','completed':len(records),'total':425,'prefixes_equal':sum(x['reference_prefix_equal'] for x in records)})
        print('EOS_GENERATED',len(records),'/425',trace['stop_reason'],'PREFIX',prefix,flush=True)
        assert prefix and same_tools,'Continuation prefix or actual tool event changed'
assert len(records)==425 and lora_hash(model)==expected
dump(out/'result.json',{'ok':True,'records':425,'all_reference_prefixes_equal':True,'all_reference_tool_events_equal':True,
 'lora_tensor_hash_unchanged':expected,'optimization_updates':0,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
