"""Continue the fixed record schedule while preserving the paused files."""
import json
from pathlib import Path
import time

from milestone_common import ROOT,PLAN,setup,seed,dump,sha,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout import rollout,tensor_sha
torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
cases=json.loads((ROOT/'inputs/cases.json').read_text())
existing=[json.loads(s) for s in (ROOT/'pause_snapshot/validated_completed_records.jsonl').read_text().splitlines()]
original=[json.loads(s) for s in (ROOT/'generation/records.jsonl').read_text().splitlines()]
assert existing==original and len(existing)==160
prefix_sha=sha(ROOT/'generation/records.jsonl')
expected=[]
for i,item in enumerate(cases):
    expected.append(f"all__{item['split']}_{item['row']}_greedy_0")
    expected.extend(f"all__{item['split']}_{item['row']}_candidate_{800000+i*4+g}" for g in range(4))
assert [x['id'] for x in existing]==expected[:160]
for x in existing:
    assert sha(ROOT/x['state_file'])==x['state_file_sha256'] and sha(ROOT/x['trace_file'])==x['trace_sha256']
out=ROOT/'generation_resume_01';out.mkdir(exist_ok=False)
checkpoint=Path(PLAN['policy']['checkpoint']);assert sha(checkpoint/'adapter_model.safetensors')==PLAN['policy']['adapter_sha256']
model,processor=load_policy(checkpoint,False);restored=restore_adapter_exact(model,checkpoint)
target_hash=PLAN['policy']['lora_tensor_hash'];assert lora_hash(model)==target_hash and all(not p.requires_grad for p in model.parameters())
settings={'policy_lora_tensor_hash':target_hash,'restoration':restored,'policy_frozen':True,'optimization_updates':0,
          'prompt':PLAN['prompt'],'inherited_records':160,'original_records_sha256':prefix_sha}
dump(out/'settings.json',settings)
images=case_images(cases[32:]);tools=real_tools();records=[];known={x['id'] for x in existing}
for i,item in enumerate(cases):
    for role,sample_seed,temp in [('greedy',0,0.)]+[('candidate',800000+i*4+g,1.) for g in range(4)]:
        stem=f"{item['split']}_{item['row']}_{role}_{sample_seed}";record_id='all__'+stem
        if record_id in known:continue
        seed(sample_seed)
        hidden,trace=rollout(model,processor,images[item['image']]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp)
        path=out/'priors'/(stem+'.pt');path.parent.mkdir(exist_ok=True);torch.save(hidden,path)
        loaded=torch.load(path,map_location='cpu',weights_only=True);assert torch.equal(loaded,hidden)
        dump(out/'traces'/(stem+'.json'),trace)
        x={'id':record_id,'image':item['image'],'split':item['split'],'row':item['row'],'image_order':i,'role':role,'seed':sample_seed,
           'temperature':temp,'quality_usable':trace['quality_usable'],'tools_used':trace['tools_used'],'stop_reason':trace['stop_reason'],
           'state_file':str(path.relative_to(ROOT)),'state_file_sha256':sha(path),'generation_state_sha256':tensor_sha(hidden),
           'cache_state_sha256':tensor_sha(loaded),'trace_file':str((out/'traces'/(stem+'.json')).relative_to(ROOT)),
           'trace_sha256':sha(out/'traces'/(stem+'.json'))}
        records.append(x)
        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
        dump(ROOT/'progress_resume.json',{'phase':'generation','mode':'resume','completed':160+len(records),'total':500,'new_records':len(records)})
        print('RESUMED',160+len(records),'/500',trace['stop_reason'],flush=True)
assert len(records)==340 and lora_hash(model)==target_hash
combined=existing+records;assert [x['id'] for x in combined]==expected
assert sha(ROOT/'generation/records.jsonl')==prefix_sha
full=ROOT/'generation_complete';full.mkdir(exist_ok=False)
(full/'records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in combined))
dump(full/'settings.json',settings)
result={'ok':True,'records':500,'inherited_records':160,'resumed_records':340,
        'quality_pass':sum(x['quality_usable'] for x in combined),'real_tool_executions':sum(x['tools_used'] for x in combined),
        'lora_tensor_hash_unchanged':target_hash,'elapsed_resume_seconds':time.monotonic()-start,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'original_records_sha256':prefix_sha}
dump(full/'result.json',result);dump(out/'result.json',{'ok':True,'records':340,'lora_tensor_hash_unchanged':target_hash})
dump(ROOT/'resume_provenance.json',{'ok':True,'original_completed_records':160,'new_records':340,
     'original_generation_records_sha256':prefix_sha,'all_original_record_fields_preserved':True,
     'original_seed_schedule_preserved':True,'merged_generation_index':'generation_complete/records.jsonl'})
