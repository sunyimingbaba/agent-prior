"""Isolated, corrected live-tool group policy-gradient training entry point."""
import argparse
import copy
import json
from pathlib import Path
import time

from milestone_common import ROOT, PLAN, dump, sha, setup, seed, table_images, load_policy, load_proxy, real_tools, lora_hash
from live_rollout import rollout, response_logprob, tensor_sha
torch=setup()
from agent.rollout import SYSTEM_PROMPT

p=argparse.ArgumentParser()
p.add_argument('--arm',choices=['smoke','retrieval','format_only'],required=True)
a=p.parse_args()
out=ROOT/'training'/a.arm
out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
is_smoke=a.arm=='smoke'
rows=PLAN['smoke_rows'] if is_smoke else PLAN['training_rows']
G=PLAN['smoke']['group_size'] if is_smoke else PLAN['training']['group_size']
lam=1. if is_smoke else PLAN['arms'][a.arm]['lambda_retrieval']
assert not set(rows)&set(PLAN['validation_rows'])
images=table_images(rows)
seed(0)
model,processor=load_policy(trainable=True)
scorer=load_proxy()
tools=real_tools()
parameters=[p for p in model.parameters() if p.requires_grad]
assert parameters and all(p.dtype==torch.float32 for p in parameters)
optimizer=torch.optim.AdamW(parameters,lr=PLAN['training']['lr'],weight_decay=PLAN['training']['adamw_weight_decay'])
initial_hash=lora_hash(model)
step_records=[];updates=0;witness=None
dump(out/'settings.json',{'arm':a.arm,'lambda_retrieval':lam,'initial_lora_tensor_hash':initial_hash,
     'trainable_parameters':sum(p.numel() for p in parameters),'lora_dtype':'float32',
     'training_rows':rows,'group_size':G,'config':PLAN['training']})

for step,row in enumerate(rows):
    item=images[row]
    samples=[]
    for g in range(G):
        sample_seed=(900000 if is_smoke else 10000)+step*G+g
        seed(sample_seed)
        hidden,trace=rollout(model,processor,item['image'],item['caption'],tools,SYSTEM_PROMPT,PLAN['rollout'])
        stem=f'{step:03d}_{row}_{g}'
        cache=out/'priors'/(stem+'.pt');cache.parent.mkdir(exist_ok=True)
        torch.save(hidden,cache)
        restored=torch.load(cache,map_location='cpu',weights_only=True)
        assert torch.equal(hidden,restored) and tensor_sha(restored)==trace['state_tensor_sha256']
        rank,ret=scorer.score(restored.float().tolist(),row)
        fmt=.2 if trace['quality_usable'] else 0.
        reward=fmt+lam*ret
        record={'id':stem,'step':step,'row':row,'image':item['name'],'sample':g,'seed':sample_seed,
            'quality_usable':trace['quality_usable'],'tools_used':trace['tools_used'],'stop_reason':trace['stop_reason'],
            'proxy_rank0':rank,'r_retrieval':ret,'r_format':fmt,'r_total':reward,
            'state_file':str(cache.relative_to(ROOT)),'state_file_sha256':sha(cache),
            'generation_state_sha256':trace['state_tensor_sha256'],'scorer_state_sha256':tensor_sha(restored),
            'trace_file':str((out/'traces'/(stem+'.json')).relative_to(ROOT))}
        dump(ROOT/record['trace_file'],trace)
        samples.append((record,trace))
        with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    if is_smoke and witness is None:
        # Separate software witness executes a real tool; it never enters training rewards.
        seed(800000)
        state,forced=rollout(model,processor,item['image'],item['caption'],tools,SYSTEM_PROMPT,
                             PLAN['rollout'],force_initial_tool=True)
        assert forced['tools_used']>=1
        dump(out/'forced_tool_software_witness.json',forced)
        eligible=next((trace for _,trace in samples if trace['rounds']),None)
        round_record=eligible['rounds'][-1]
        model.eval()
        with torch.no_grad():
            lp,counts=response_logprob(model,processor,item['image'],round_record)
            padded=copy.deepcopy(round_record)
            padded['generated_ids'] += [processor.tokenizer.pad_token_id]*17
            lp_padded,pad_counts=response_logprob(model,processor,item['image'],padded)
        assert float(lp)==float(lp_padded) and counts['effective_tokens']==pad_counts['effective_tokens']
        witness={'actual_forced_tool_executions':forced['tools_used'],'software_only_witness':True,
            'padding_logprob_exact_equal':True,'original_logprob':float(lp),'padded_logprob':float(lp_padded),
            'effective_tokens':counts['effective_tokens'],'added_padding_ignored':pad_counts['post_generation_padding_ignored'],
            'state_source':'unchanged generation hidden tensor cached before any logprob forward'}
        dump(out/'contract_witness.json',witness)
    rewards=torch.tensor([x['r_total'] for x,_ in samples],device='cuda',dtype=torch.float32)
    std=rewards.std()
    active=bool(std>1e-8)
    record={'step':step,'row':row,'reward_mean':float(rewards.mean()),'reward_sample_std':float(std),
            'quality_count':sum(x['quality_usable'] for x,_ in samples),'actual_tools':sum(x['tools_used'] for x,_ in samples),
            'optimizer_update':active,'effective_loss_tokens':0,'gradient_norm_before_clip':0.,'loss':0.}
    if active:
        advantages=(rewards-rewards.mean())/(std+1e-8)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for g,(_,trace) in enumerate(samples):
            for round_record in trace['rounds']:
                logp,counts=response_logprob(model,processor,item['image'],round_record)
                loss=-advantages[g].detach()*logp/G
                assert torch.isfinite(loss)
                loss.backward()
                record['loss']+=float(loss.detach())
                record['effective_loss_tokens']+=counts['effective_tokens']
                del logp,loss
        norm=torch.nn.utils.clip_grad_norm_(parameters,PLAN['training']['gradient_clip_norm'])
        assert torch.isfinite(norm) and float(norm)>0
        record['gradient_norm_before_clip']=float(norm)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        optimizer.step();updates+=1
        model.eval()
    assert all(not p.requires_grad and p.grad is None for p in scorer.pl.parameters())
    step_records.append(record)
    dump(out/'steps.json',step_records)
    dump(ROOT/'progress.json',{'phase':'training','arm':a.arm,'attempted_steps':step+1,'total':len(rows),
                               'optimizer_updates':updates,'last_quality_count':record['quality_count']})
    print('TRAIN',a.arm,step+1,'/',len(rows),'updates',updates,'quality',record['quality_count'],'/',G,flush=True)
    if is_smoke and updates==1:break

final_hash=lora_hash(model)
if is_smoke:
    assert updates==1 and final_hash!=initial_hash
else:
    checkpoint=out/'checkpoint'
    model.save_pretrained(checkpoint,safe_serialization=True)
    processor.save_pretrained(checkpoint)
    dump(out/'checkpoint_manifest.json',{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size}
                                       for p in checkpoint.iterdir() if p.is_file()})
result={'ok':True,'arm':a.arm,'attempted_steps':len(step_records),'optimizer_updates':updates,
        'initial_lora_tensor_hash':initial_hash,'final_lora_tensor_hash':final_hash,
        'parameters_changed':initial_hash!=final_hash,'quality_pass_samples':sum(x['quality_count'] for x in step_records),
        'total_samples':len(step_records)*G,'real_tool_executions':sum(x['actual_tools'] for x in step_records),
        'frozen_scorer_unchanged_requires_grad':True,'elapsed_seconds':time.monotonic()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
dump(out/'result.json',result)
print(json.dumps(result),flush=True)
