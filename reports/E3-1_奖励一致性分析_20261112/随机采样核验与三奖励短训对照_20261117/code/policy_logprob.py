import torch

def effective_response(ids,eos_ids,recorded_length):
    assert len(ids)==recorded_length and recorded_length>0
    hits=[i for i,x in enumerate(ids) if x in eos_ids]
    assert not hits or hits==[len(ids)-1]
    return list(ids)

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
