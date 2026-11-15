import torch
from live_rollout_eos import rollout,effective_response,tensor_sha

def recorded_rollout(model,processor,image,caption,tools,system,config,capture=False):
    original=model.generate;calls=[];witness={}
    def generate(*args,**kw):
        kw.update(output_scores=True,output_logits=True)
        output=original(*args,**kw);p=kw['input_ids'].shape[1];ids=output.sequences[0,p:]
        scores=torch.stack(output.scores)[:,0,:].float();raw=torch.stack(output.logits)[:,0,:].float()
        eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
        expected=raw.clone();expected[0,eos]=-torch.inf;assert torch.equal(expected,scores)
        lp=torch.log_softmax(scores,dim=-1).gather(1,ids[:,None]).flatten()
        row={'actual_generation_token_logprobs':lp.cpu().tolist(),'policy_processor_exact':True}
        if capture:
            pos=sorted({0,len(ids)//2,len(ids)-1});witness[len(calls)]={'positions':pos,'generation_logits':scores[pos].cpu(),'targets':ids[pos].cpu(),'eos_ids':eos}
        calls.append(row);output.scores=None;output.logits=None
        return output
    model.generate=generate
    try:h,trace=rollout(model,processor,image,caption,tools,system,config)
    finally:model.generate=original
    assert len(calls)==len(trace['rounds'])
    for x,extra in zip(trace['rounds'],calls):x.update(extra)
    return h,trace,witness

def grad_logprob(model,processor,image,record,capture=False):
    ids=effective_response(record['generated_ids'],record['eos_token_ids'],record['generated_token_count'])
    inputs=processor(text=[record['prompt_text']],images=[image],return_tensors='pt').to('cuda');p=inputs.input_ids.shape[1]
    assert inputs.input_ids[0].tolist()==record['prompt_token_ids']
    target=torch.tensor(ids,device='cuda',dtype=torch.long);assert ids[0] not in record['eos_token_ids']
    inputs['input_ids']=torch.cat([inputs.input_ids,target[None,:-1]],dim=1);inputs['attention_mask']=torch.ones_like(inputs.input_ids)
    out=model(**inputs,use_cache=False,return_dict=True,output_hidden_states=capture)
    logits=out.logits[0,p-1:p+len(ids)-1].float().clone();assert len(logits)==len(ids)
    logits[0,record['eos_token_ids']]=-torch.inf
    lps=torch.log_softmax(logits,dim=-1).gather(1,target[:,None]).flatten();value=lps.sum()
    detail={'effective_tokens':len(ids),'post_generation_padding_ignored':len(record['generated_ids'])-len(ids),'forward_input_tokens':int(inputs.input_ids.shape[1]),'token_logprobs':lps.detach().cpu().tolist()}
    extra={}
    if capture:
        pos=sorted({0,len(ids)//2,len(ids)-1});extra={'positions':pos,'teacher_logits':logits[pos].detach().cpu(),'teacher_hidden':out.hidden_states[-1][0,-1].detach().cpu()}
    return value,detail,extra
