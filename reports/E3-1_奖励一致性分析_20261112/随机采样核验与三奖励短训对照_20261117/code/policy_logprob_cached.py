"""Differentiable one-token KV-cache replay matching the actual generate forward geometry."""
import torch
from transformers.cache_utils import DynamicCache
from policy_logprob import effective_response

def grad_logprob(model,processor,image,record,capture=False):
 ids=effective_response(record['generated_ids'],record['eos_token_ids'],record['generated_token_count']);inputs=processor(text=[record['prompt_text']],images=[image],return_tensors='pt').to('cuda');n=inputs.input_ids.shape[1];assert inputs.input_ids[0].tolist()==record['prompt_token_ids'];assert ids[0] not in record['eos_token_ids'];core=model.base_model.model;prior=model.training;model.eval();full_ids=inputs.pop('input_ids');kw={**dict(inputs),'use_cache':True,'past_key_values':DynamicCache(config=core.config),'cache_position':torch.arange(n,device='cuda'),'logits_to_keep':1};lps=[];rows=[];positions=sorted({0,len(ids)//2,len(ids)-1});last=None
 for i,token in enumerate(ids):
  prepared=core.prepare_inputs_for_generation(full_ids,**kw);out=core(**prepared,return_dict=True,output_hidden_states=capture);logits=out.logits[0,-1].float().clone()
  if i==0:logits[record['eos_token_ids']]=-torch.inf
  lps.append(torch.log_softmax(logits,dim=-1)[token])
  if capture and i in positions:rows.append(logits.detach().cpu())
  if capture and i==len(ids)-1:last=out.hidden_states[-1][0,-1].detach().cpu()
  kw=core._update_model_kwargs_for_generation(out,kw);full_ids=torch.cat([full_ids,torch.tensor([[token]],device=full_ids.device)],dim=1)
  del out,prepared,logits
 values=torch.stack(lps);total=values.sum();detail={'effective_tokens':len(ids),'post_generation_padding_ignored':0,'forward_input_tokens':n+len(ids)-1,'forward_geometry':'native prepare_inputs_for_generation, prefill thenone-token DynamicCache, logits_to_keep1, evalmode dropout0, autograd enabled','token_logprobs':values.detach().cpu().tolist()};extra={'positions':positions,'teacher_logits':torch.stack(rows),'teacher_hidden':last} if capture else {};model.train(prior);return total,detail,extra
