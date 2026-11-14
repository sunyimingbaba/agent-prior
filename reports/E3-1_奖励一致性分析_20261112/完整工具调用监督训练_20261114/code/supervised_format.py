"""Causal format-token targets with body labels and prompt positions excluded."""
import torch
import torch.nn.functional as F

def answer_mask(offsets,prefix_end,body_end):
    return [bool(end>start and (end<=prefix_end or start>=body_end)) for start,end in offsets]

def target_tokens(tokenizer,kind,body=''):
    if kind=='call':
        text='<tool_call>scene_classify</tool_call>'
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        mask=[True]*len(encoded['input_ids'])
    else:
        assert kind=='answer' and body
        text='<answer>'+body+'</answer>'
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        mask=answer_mask(encoded['offset_mapping'],len('<answer>'),len('<answer>')+len(body))
    eos=tokenizer.convert_tokens_to_ids('<|im_end|>');assert isinstance(eos,int)
    ids=encoded['input_ids']+[eos];mask=mask+[True]
    assert ids and sum(mask)>1 and tokenizer.decode(ids[:-1],skip_special_tokens=False)==text
    return {'kind':kind,'text':text,'target_ids':ids,'loss_mask':mask,
            'offsets':[list(x) for x in encoded['offset_mapping']]+[[-1,-1]],
            'body_char_span':[len('<answer>'),len('<answer>')+len(body)] if kind=='answer' else None,
            'supervised_tokens':sum(mask),'total_target_tokens':len(ids),'eos_id':eos}

def masked_token_nll(logits,target,mask,eos_ids,temperature=1.):
    assert logits.ndim==2 and logits.shape[0]==target.shape[0]==mask.shape[0]
    assert temperature>0 and target[0].item() not in eos_ids
    scaled=logits.float()/temperature
    scaled=scaled.clone();scaled[0,eos_ids]=-torch.inf
    # Only selected response positions contribute a cross-entropy term.
    nll=F.cross_entropy(scaled[mask],target[mask],reduction='none')
    selected_logits=scaled[mask];pred=selected_logits.argmax(dim=-1)
    return nll.sum(),{'active_tokens':int(mask.sum()),'nll_per_active_token':nll.detach().cpu().tolist(),
       'target_ids_active':target[mask].detach().cpu().tolist(),'argmax_ids_active':pred.detach().cpu().tolist(),
       'active_argmax_correct':int((pred==target[mask]).sum()),'temperature':temperature,
       'first_token_eos_suppressed':True}

def response_loss(model,processor,image,example,eos_ids):
    rendered=example['prompt_text'];inputs=processor(text=[rendered],images=[image],return_tensors='pt').to('cuda')
    assert inputs.input_ids[0].tolist()==example['prompt_token_ids']
    target=torch.tensor(example['target_ids'],device='cuda',dtype=torch.long)
    mask=torch.tensor(example['loss_mask'],device='cuda',dtype=torch.bool)
    p=inputs.input_ids.shape[1];full=torch.cat([inputs.input_ids,target[None]],dim=1)
    inputs['input_ids']=full;inputs['attention_mask']=torch.ones_like(full)
    out=model(**inputs,use_cache=False,return_dict=True)
    logits=out.logits[0,p-1:p+len(target)-1]
    value,stats=masked_token_nll(logits,target,mask,eos_ids,temperature=1.)
    stats.update(prompt_tokens=p,response_tokens=len(target),forward_tokens=int(full.shape[1]),
                 prompt_loss_tokens=0,body_loss_tokens=0)
    return value,stats
