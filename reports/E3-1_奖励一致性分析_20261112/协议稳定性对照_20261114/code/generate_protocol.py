import json
from pathlib import Path
import time
from milestone_common import ROOT,PLAN,setup,seed,dump,sha,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout_legacy import rollout as legacy_rollout,tensor_sha
from live_rollout_protocol import rollout,STRICT_TOOL,STRICT_ANSWER

torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
cases=json.loads((ROOT/'inputs/cases.json').read_text());by={x['image']:x for x in cases}
order={x['image']:i for i,x in enumerate(cases)};images=case_images(cases);tools=real_tools()
prompts=json.loads((ROOT/'inputs/protocol_prompts.json').read_text())
model,processor=load_policy(Path(PLAN['policy']['checkpoint']),False)
restored=restore_adapter_exact(model,Path(PLAN['policy']['checkpoint']));policy_hash=PLAN['policy']['lora_tensor_hash']
assert lora_hash(model)==policy_hash and all(not p.requires_grad for p in model.parameters())
dump(ROOT/'policy_settings.json',{'policy_frozen':True,'optimization_updates':0,'lora_tensor_hash':policy_hash,'restoration':restored})
all_records=[]

def generate_one(arm,item,role,sample_seed,folder,legacy=False):
    seed(sample_seed);temp=0. if role=='greedy' else arm['temperature']
    args=(model,processor,images[item['image']]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'])
    hidden,trace=legacy_rollout(*args,temperature=temp) if legacy else rollout(*args,temperature=temp,
        prompt_mode=arm['prompt_mode'],protocol_prompts=prompts)
    answer=STRICT_ANSWER.fullmatch(trace['rounds'][-1]['generated_text'].strip())
    strict=bool(trace['quality_usable'] and answer and answer.group(1).strip() and
                all(STRICT_TOOL.fullmatch(z['generated_text'].strip()) for z in trace['rounds'][:-1]))
    stem=f"{arm['name']}__{item['split']}_{item['row']}_{role}_{sample_seed}"
    out=ROOT/folder;out.mkdir(exist_ok=True);path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True)
    torch.save(hidden,path);loaded=torch.load(path,map_location='cpu',weights_only=True);assert torch.equal(hidden,loaded)
    trace_path=out/'traces'/(stem+'.json');dump(trace_path,trace)
    x={'id':stem,'arm':arm['name'],'image':item['image'],'split':item['split'],'row':item['row'],
       'image_order':order[item['image']],'role':role,'seed':sample_seed,'temperature':temp,
       'quality_usable':trace['quality_usable'],'strict_protocol_usable':strict,'tools_used':trace['tools_used'],
       'stop_reason':trace['stop_reason'],'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),
       'generation_state_sha256':tensor_sha(hidden),'cache_state_sha256':tensor_sha(loaded),
       'trace_file':str(trace_path.relative_to(ROOT)),'trace_sha256':sha(trace_path)}
    with (out/'records.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    all_records.append(x);dump(ROOT/'progress.json',{'phase':'generation','completed_unique':len(all_records),'total_unique':626,
          'arm':arm['name'],'image':item['image']})
    print('GENERATED',len(all_records),'/626',arm['name'],trace['stop_reason'],flush=True)
    return x,trace

assert not (ROOT/'baseline_reproduction').exists() and not (ROOT/'pilot').exists()
baseline=json.loads((ROOT/'baseline_evidence/scores.json').read_text());reproduction=[]
legacy_arm={'name':'legacy_reproduction','temperature':1.,'prompt_mode':'legacy'}
for name in PLAN['baseline_reproduction']['images']:
    item=by[name];i=order[name]
    for role,sample_seed in [('greedy',0),('candidate',800000+i*4),('candidate',800000+i*4+1)]:
        old=next(x for x in baseline if x['image']==name and x['role']==role and x['seed']==sample_seed)
        x,t=generate_one(legacy_arm,item,role,sample_seed,'baseline_reproduction',legacy=True)
        ot=json.loads((Path(PLAN['baseline_root'])/old['trace_file']).read_text())
        assert [z['generated_ids'] for z in t['rounds']]==[z['generated_ids'] for z in ot['rounds']]
        assert t['stop_reason']==ot['stop_reason'] and t['tool_events']==ot['tool_events']
        assert x['generation_state_sha256']==old['generation_state_sha256']
        reproduction.append({'new_id':x['id'],'baseline_id':old['id'],'emitted_token_ids_equal':True,
                             'all_tool_events_equal':True,'hidden_tensor_sha256_equal':True})
dump(ROOT/'baseline_reproduction/result.json',{'ok':True,'records':6,'checks':reproduction})
pilot=[];stats=[]
for arm in PLAN['arms']:
    records=[]
    for item in cases:
        if item['image'] not in PLAN['calibration_images']:continue
        for g in range(4):records.append(generate_one(arm,item,'candidate',800000+order[item['image']]*4+g,'pilot/'+arm['name'])[0])
    assert len(records)==60;pilot+=records
    diverse=sum(len({x['generation_state_sha256'] for x in records if x['image']==name and x['quality_usable']})>=2
                for name in PLAN['calibration_images'])
    quality=sum(x['quality_usable'] for x in records)/60;strict=sum(x['strict_protocol_usable'] for x in records)/60
    result_rate=sum(x['stop_reason']=='model_generated_tool_result' for x in records)/60
    gate=PLAN['pilot_selection'];eligible=quality>=gate['legacy_quality_min'] and strict>=gate['strict_quality_min'] \
      and result_rate<=gate['generated_result_failure_rate_max'] and diverse>=gate['images_with_2_valid_distinct_states_min']
    stats.append({'arm':arm['name'],'quality':quality,'strict_quality':strict,'generated_result_failure_rate':result_rate,
                  'images_with_2_valid_distinct_states':diverse,'eligible':eligible})
    dump(ROOT/'pilot_progress.json',stats)
eligible=[s for s in stats if s['eligible']]
chosen=eligible[0] if eligible else sorted(stats,key=lambda x:(-x['quality'],x['generated_result_failure_rate'],
    [a['name'] for a in PLAN['arms']].index(x['arm'])))[0]
selected=next(a for a in PLAN['arms'] if a['name']==chosen['arm'])
dump(ROOT/'protocol_selection.json',{'ok':True,'used_calibration_train_only':True,'pilot_criteria_satisfied':bool(eligible),
     'selected_arm':selected,'pilot_stats':stats,'policy_lora_tensor_hash':policy_hash})
print('PROTOCOL_SELECTED',json.dumps(selected),flush=True)
full=[];pilot_lookup={(x['image'],x['seed']):x for x in pilot if x['arm']==selected['name']}
for item in cases:
    full.append(generate_one(selected,item,'greedy',0,'confirmation')[0])
    for g in range(4):
        s=800000+order[item['image']]*4+g
        full.append(pilot_lookup.get((item['image'],s)) or generate_one(selected,item,'candidate',s,'confirmation')[0])
assert len(full)==500 and len(all_records)==626 and lora_hash(model)==policy_hash
out=ROOT/'generation_complete';out.mkdir()
(out/'records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in full))
(ROOT/'all_generation_records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in all_records))
dump(out/'result.json',{'ok':True,'records':500,'all_unique_records':626,'selected_arm':selected,
   'lora_tensor_hash_unchanged':policy_hash,'optimization_updates':0,'elapsed_seconds':time.monotonic()-start,
   'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
print('ALL_PROTOCOL_GENERATION_COMPLETE',flush=True)
