"""Unchanged native-EOS controller, saving final-layer states for EVERY actual decode step."""
import hashlib
import re
import torch
from state_contract import body_layout,extract_states
TOOL=re.compile(r'<tool_call>\s*scene_classify\s*</tool_call>')
ANSWER=re.compile(r'<answer>(.*?)</answer>',re.S)
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
class ToolOnlyStop:
 def __init__(self,processor,p):self.processor,self.p=processor,p
 def __call__(self,ids,scores,**kwargs):return torch.tensor([bool(TOOL.fullmatch(self.processor.decode(ids[0,self.p:].tolist(),skip_special_tokens=True).strip()))],device=ids.device)

@torch.no_grad()
def rollout(model,processor,image,messages,feedback,config):
 messages=list(messages);rounds=[];states=[];events=[];stop='round_limit';prior=model.training;model.eval()
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
 for ri in range(config['max_rounds']):
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);assert text.endswith('<|im_start|>assistant\n')
  inputs=processor(text=[text],images=[image],return_tensors='pt').to('cuda');p=inputs.input_ids.shape[1]
  kwargs=dict(max_new_tokens=config['max_new_tokens_per_round'],min_new_tokens=1,do_sample=False,temperature=1.,top_k=0,top_p=1.,repetition_penalty=1.,use_cache=True,output_hidden_states=True,return_dict_in_generate=True)
  out=model.generate(**inputs,**kwargs,stopping_criteria=[ToolOnlyStop(processor,p)])
  ids=out.sequences[0,p:].cpu().tolist();assert len(ids)==len(out.hidden_states)>0
  hs=torch.stack([v[-1][0,-1,:].detach().cpu().contiguous() for v in out.hidden_states]);assert hs.shape==(len(ids),3584) and hs.dtype==torch.bfloat16 and torch.isfinite(hs).all()
  generated=processor.decode(ids,skip_special_tokens=True);tokens=processor.tokenizer.convert_ids_to_tokens(ids);layout=body_layout(ids,tokens,eos,generated)
  limited=len(ids)==config['max_new_tokens_per_round'];rounds.append({'round':ri+1,'prompt_text':text,'prompt_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'message_roles':[m['role'] for m in messages],'prompt_token_ids':inputs.input_ids[0].cpu().tolist(),'generated_ids':ids,'generated_token_strings':tokens,'generated_text':generated,'generated_token_count':len(ids),'token_limit':limited,'eos_token_ids':eos,'state_tensor_sha256':tensor_sha(hs[-1]),'step_state_numeric_sha256':tensor_sha(hs),'body_layout':layout,'generation_kwargs':{k:v for k,v in kwargs.items() if k not in ['output_hidden_states','return_dict_in_generate']}});states.append(hs)
  del out,inputs
  if TOOL.fullmatch(generated.strip()) and not limited:
   event={**feedback,'name':'scene_classify','after_round':ri+1,'origin':'model_requested_recorded_real_feedback'};events.append(event)
   messages += [{'role':'assistant','content':generated},{'role':'user','content':'<result>'+feedback['text']+'</result>'}];continue
  if limited:stop='token_limit'
  elif '<result>' in generated or '</result>' in generated:stop='model_generated_tool_result'
  elif '<tool_call>' in generated:stop='mixed_or_unknown_tool_output'
  elif ANSWER.search(generated) and ANSWER.search(generated).group(1).strip():stop='answer'
  else:stop='format_failure' if generated.strip() else 'empty'
  break
 model.train(prior);quality=bool(stop=='answer' and all(not q['token_limit'] for q in rounds) and all('addCriterion' not in q['generated_text'] for q in rounds))
 strict=bool(quality and re.fullmatch(r'<answer>\s*.+?\s*</answer>',rounds[-1]['generated_text'].strip(),re.S) and all(TOOL.fullmatch(q['generated_text'].strip()) for q in rounds[:-1]))
 candidates,coverage=extract_states(rounds,states,events,strict)
 trace={'rounds':rounds,'tool_events':events,'tools_used':len(events),'stop_reason':stop,'quality_usable':quality,'strict_protocol_usable':strict,'trajectory_text':'\n'.join(q['generated_text'] for q in rounds),'state_rule':'last actual generate final-layer last valid decode-step; emitted token j is processed at step j+1','state_tensor_sha256':tensor_sha(candidates['last']),'state_coverage':coverage,'software_forced_tool':False}
 return candidates,states,trace
