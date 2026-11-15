import copy
import json
from pathlib import Path
import time
from milestone_common import ROOT,PLAN,setup,seed,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from quantity_model import QuantityScorer
from training_contract import recorded_rollout,grad_logprob
from live_rollout_eos import rollout,tensor_sha
from stage_io import read,dump,sha

torch=setup();seed(0);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
cases=read(ROOT/'inputs/evaluation_cases.json');allcases=read(ROOT/'inputs/cases.json');images=case_images(cases);tools=real_tools()
model,processor=load_policy(Path(PLAN['policy']['checkpoint']),True);restore_adapter_exact(model,Path(PLAN['policy']['checkpoint']))
expected=PLAN['policy']['lora_tensor_hash'];assert lora_hash(model)==expected
scorer=QuantityScorer(allcases);gate=ROOT/'contract_gate';gate.mkdir();name=PLAN['training_images'][0];image=images[name]['image'];witness={};rows=[];traces=[]
for g in range(2):
    seed(1700000+g);h,tr,ws=recorded_rollout(model,processor,image,PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],capture=True)
    torch.save(h,gate/f'state_{g}.pt');dump(gate/f'trace_{g}.json',tr);traces.append(tr);model.eval()
    for ri,z in enumerate(tr['rounds']):
        with torch.no_grad():
            lp,detail,extra=grad_logprob(model,processor,image,z,capture=True)
            padded=copy.deepcopy(z);padded['generated_ids'] += [processor.tokenizer.pad_token_id]*17
            plp,pdetail,_=grad_logprob(model,processor,image,padded)
        assert float(lp)==float(plp) and pdetail['post_generation_padding_ignored']==17
        diff=torch.tensor(detail['token_logprobs'])-torch.tensor(z['actual_generation_token_logprobs']);assert float(diff.abs().max())<=PLAN['probability_gate']['teacher_vs_generation_max_token_logprob_gap'] and float(diff.abs().mean())<=PLAN['probability_gate']['teacher_vs_generation_mean_absolute_token_logprob_gap']
        if ri==len(tr['rounds'])-1:
            cos=float(torch.nn.functional.cosine_similarity(h.float(),extra['teacher_hidden'].float(),dim=0));assert cos>=PLAN['probability_gate']['teacher_state_cosine_min']
        else:cos=None
        witness[f'{g}_{ri}']={**ws[ri],**extra,'generation_chosen_logprobs':[z['actual_generation_token_logprobs'][i] for i in ws[ri]['positions']],
          'teacher_chosen_logprobs':[detail['token_logprobs'][i] for i in ws[ri]['positions']]}
        rows.append({'sample':g,'round':ri,'logprob':float(lp),'effective_tokens':detail['effective_tokens'],'padding_ignored':True,
          'max_logprob_gap':float(diff.abs().max()),'mean_abs_logprob_gap':float(diff.abs().mean()),'final_teacher_state_cosine':cos})
    value=scorer.score(h,next(x for x in cases if x['image']==name));assert tensor_sha(h)==tr['state_tensor_sha256']
torch.save(witness,gate/'logit_witness.pt');assert sum(x['tools_used'] for x in traces)>0
parameters=[p for p in model.parameters() if p.requires_grad];model.train();temporary=torch.optim.AdamW(parameters,lr=PLAN['training']['lr'],weight_decay=PLAN['training']['weight_decay'])
temporary.zero_grad(set_to_none=True);denominator=sum(z['generated_token_count'] for z in traces[0]['rounds'])
for z in traces[0]['rounds']:
    lp,_,_=grad_logprob(model,processor,image,z);(-lp/denominator).backward()
norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.));assert norm>0 and torch.isfinite(torch.tensor(norm))
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
temporary.step();changed=lora_hash(model);assert changed!=expected
free,total=torch.cuda.mem_get_info();headroom=free/1024**2;assert headroom>=PLAN['probability_gate']['memory_reserved_headroom_MiB_min']
memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'free_after_Adam_MiB':headroom,'total_MiB':total/1024**2}
temporary.zero_grad(set_to_none=True);del temporary;restore_adapter_exact(model,Path(PLAN['policy']['checkpoint']));assert lora_hash(model)==expected
dump(gate/'result.json',{'ok':True,'probability_rows':rows,'exact_eos_processor_verified':True,'real_tools':sum(x['tools_used'] for x in traces),'gradient_norm':norm,
  'temporary_optimizer_steps':1,'permanent_optimizer_updates':0,'temporary_hash':changed,'restored_initial_hash':expected,'memory':memory})
del scorer;model.eval();model.requires_grad_(False);torch.cuda.empty_cache()
out=ROOT/'before';out.mkdir();records=[];order={x['image']:i for i,x in enumerate(allcases)}
for item in cases:
    for role,sample_seed,temp in [('greedy',0,0.)]+[('candidate',800000+order[item['image']]*4+k,1.) for k in range(4)]:
        seed(sample_seed);h,tr=rollout(model,processor,images[item['image']]['image'],PLAN['prompt'],tools,SYSTEM_PROMPT,PLAN['rollout'],temperature=temp)
        stem=f"before__{item['split']}_{item['row']}_{role}_{sample_seed}";sf=out/'priors'/(stem+'.pt');sf.parent.mkdir(exist_ok=True);torch.save(h,sf)
        tf=out/'traces'/(stem+'.json');dump(tf,tr)
        row={'id':stem,'arm':'before','image':item['image'],'split':item['split'],'row':item['row'],'role':role,'seed':sample_seed,'temperature':temp,
          'quality_usable':tr['quality_usable'],'strict_protocol_usable':tr['strict_protocol_usable'],'tools_used':tr['tools_used'],'stop_reason':tr['stop_reason'],
          'state_file':str(sf.relative_to(ROOT)),'state_file_sha256':sha(sf),'generation_state_sha256':tensor_sha(h),'trace_file':str(tf.relative_to(ROOT)),'trace_sha256':sha(tf)}
        records.append(row)
        with (out/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        dump(ROOT/'progress.json',{'phase':'before_generation','completed':len(records),'total':425});print('RTX_BASELINE',len(records),'/425',flush=True)
assert len(records)==425 and lora_hash(model)==expected
dump(out/'result.json',{'ok':True,'records':425,'policy_hash_unchanged':expected,'permanent_optimizer_updates':0,'elapsed_seconds':time.monotonic()-start})
