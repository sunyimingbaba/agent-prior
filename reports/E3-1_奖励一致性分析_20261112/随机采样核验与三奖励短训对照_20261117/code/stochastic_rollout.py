"""Actual on-policy sampling, nativeEOS, model-requested feedback and per-step evidence."""
import hashlib
import re
import torch
from state_contract_v2 import body_layout
TOOL=re.compile(r'<tool_call>\s*scene_classify\s*</tool_call>')
ANSWER=re.compile(r'<answer>(.*?)</answer>',re.S)
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
class ToolOnlyStop:
 def __init__(self,processor,p):self.processor,self.p=processor,p
 def __call__(self,ids,scores,**kwargs):return torch.tensor([bool(TOOL.fullmatch(self.processor.decode(ids[0,self.p:].tolist(),skip_special_tokens=True).strip()))],device=ids.device)

@torch.no_grad()
def rollout(model,processor,image,messages,feedback,config,temperature=1.,capture=False):
 messages=list(messages);rounds=[];states=[];events=[];witness={};stop='round_limit';prior=model.training;model.eval()
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
 for ri in range(config['max_rounds']):
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);assert text.endswith('<|im_start|>assistant\n')
  inputs=processor(text=[text],images=[image],return_tensors='pt').to('cuda');p=inputs.input_ids.shape[1]
  kwargs=dict(max_new_tokens=config['max_new_tokens_per_round'],min_new_tokens=1,do_sample=temperature>0,temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.,use_cache=True,output_hidden_states=True,output_scores=True,output_logits=True,return_dict_in_generate=True)
  out=model.generate(**inputs,**kwargs,stopping_criteria=[ToolOnlyStop(processor,p)])
  ids=out.sequences[0,p:].cpu().tolist();assert len(ids)==len(out.hidden_states)>0
  hs=torch.stack([v[-1][0,-1,:].detach().cpu().contiguous() for v in out.hidden_states]);assert hs.shape==(len(ids),3584) and hs.dtype==torch.bfloat16 and torch.isfinite(hs).all()
  scores=torch.stack(out.scores)[:,0,:].float();raw=torch.stack(out.logits)[:,0,:].float();expected=raw.clone();expected[0,eos]=-torch.inf;assert torch.equal(expected,scores),'unexpected sampling processor'
  targets=torch.tensor(ids,device=scores.device);lp=torch.log_softmax(scores,dim=-1).gather(1,targets[:,None]).flatten().cpu().tolist()
  if capture:
   pos=sorted({0,len(ids)//2,len(ids)-1});witness[ri]={'positions':pos,'generation_logits':scores[pos].cpu(),'targets':targets[pos].cpu(),'eos_ids':eos}
  generated=processor.decode(ids,skip_special_tokens=True);tokens=processor.tokenizer.convert_ids_to_tokens(ids);limited=len(ids)==config['max_new_tokens_per_round']
  rounds.append({'round':ri+1,'prompt_text':text,'prompt_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'message_roles':[m['role'] for m in messages],'prompt_token_ids':inputs.input_ids[0].cpu().tolist(),'generated_ids':ids,'generated_token_strings':tokens,'generated_text':generated,'generated_token_count':len(ids),'token_limit':limited,'eos_token_ids':eos,'state_tensor_sha256':tensor_sha(hs[-1]),'step_state_numeric_sha256':tensor_sha(hs),'body_layout':body_layout(ids,tokens,eos,generated),'actual_generation_token_logprobs':lp,'policy_processor_exact':True,'generation_kwargs':{k:v for k,v in kwargs.items() if not k.startswith('output_') and k!='return_dict_in_generate'}});states.append(hs)
  del out,inputs,scores,raw,expected,targets
  if TOOL.fullmatch(generated.strip()) and not limited:
   events.append({**feedback,'name':'scene_classify','after_round':ri+1,'origin':'model_requested_recorded_real_feedback'})
   messages += [{'role':'assistant','content':generated},{'role':'user','content':'<result>'+feedback['text']+'</result>'}];continue
  if limited:stop='token_limit'
  elif '<result>' in generated or '</result>' in generated:stop='model_generated_tool_result'
  elif '<tool_call>' in generated or '</tool_call>' in generated:stop='mixed_or_unknown_tool_output'
  elif ANSWER.search(generated) and ANSWER.search(generated).group(1).strip():stop='answer'
  else:stop='format_failure' if generated.strip() else 'empty'
  break
 model.train(prior);result_tag=any('<result>' in x['generated_text'] or '</result>' in x['generated_text'] for x in rounds);final=rounds[-1]['generated_text'];m=ANSWER.fullmatch(final.strip())
 strict=bool(stop=='answer' and not result_tag and m and m.group(1).strip() and final.count('<answer>')==final.count('</answer>')==1 and '<tool_call>' not in final and '</tool_call>' not in final and all(not q['token_limit'] and 'addCriterion' not in q['generated_text'] for q in rounds) and all(TOOL.fullmatch(q['generated_text'].strip()) for q in rounds[:-1]))
 consumed=any(e['after_round']<len(rounds) for e in events);usable=bool(strict and consumed and len(events)>0)
 last=states[-1][-1].contiguous();trace={'rounds':rounds,'tool_events':events,'tools_used':len(events),'feedback_consumed':consumed,'stop_reason':stop,'quality_usable':usable,'strict_protocol_usable':strict,'strict_completed_consumed_tool':usable,'generated_result_tag':result_tag,'state_tensor_sha256':tensor_sha(last),'software_forced_tool':False,'temperature':temperature,'state_rule':'last actual generate final-layer final-position state, same tensor saved and scored'}
 return last,{'prior':last,'step_states':states,'probability_witness':witness},trace
