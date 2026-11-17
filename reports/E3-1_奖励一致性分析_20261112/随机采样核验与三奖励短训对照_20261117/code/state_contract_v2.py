"""Map ACTUAL emitted byte-level tokens to their actual next-step input states."""
import re
import numpy as np
import torch

ANSWER=re.compile(r'<answer>(.*?)</answer>',re.S)
STRICT_ANSWER=re.compile(r'<answer>(.*?)</answer>',re.S)
def byte_decoder():
 bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs.copy();n=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+n);n+=1
 return dict(zip(map(chr,cs),bs))

def body_layout(ids,tokens,eos_ids,text):
 """Return body-token and generate-step indices; each step j>0 consumes token j-1."""
 eos_positions=[i for i,x in enumerate(ids) if x in eos_ids]
 if eos_positions and eos_positions!=[len(ids)-1]:return {'available':False,'reason':'unexpected_eos_positions'}
 active=len(ids)-bool(eos_positions);decode=byte_decoder();parts=[];spans=[];offset=0
 for token in tokens[:active]:
  if any(c not in decode for c in token):return {'available':False,'reason':'unexpected_non_byte_token'}
  data=bytes(decode[c] for c in token);parts.append(data);spans.append([offset,offset+len(data)]);offset+=len(data)
 raw=b''.join(parts)
 try:exact=raw.decode('utf-8')
 except UnicodeDecodeError:return {'available':False,'reason':'incomplete_utf8'}
 if exact!=text:return {'available':False,'reason':'byte_decode_mismatch'}
 if text.count('<answer>')!=1 or text.count('</answer>')!=1:return {'available':False,'reason':'multiple_or_unbalanced_answer_blocks'}
 if not STRICT_ANSWER.fullmatch(text.strip()):return {'available':False,'reason':'not_strict_answer'}
 match=ANSWER.search(text);lo,hi=match.span(1);a=len(text[:lo].encode());b=len(text[:hi].encode())
 token_indices=[i for i,(s,e) in enumerate(spans) if e>s and s>=a and e<=b and i+1<len(ids)]
 if not token_indices:return {'available':False,'reason':'no_wholly_body_input_token'}
 return {'available':True,'reason':None,'body_char_span':[lo,hi],'body_byte_span':[a,b],'response_token_byte_spans':spans,'body_token_indices':token_indices,'body_state_step_indices':[i+1 for i in token_indices],'active_response_tokens':active}

def extract_states(rounds,step_states,tools,eligible):
 last=step_states[-1][-1].contiguous();received=any(e['after_round']<len(rounds) for e in tools)
 valid=bool(eligible and STRICT_ANSWER.fullmatch(rounds[-1]['generated_text'].strip()) and received and not rounds[-1]['token_limit'] and len(rounds)>1)
 layout=rounds[-1]['body_layout'];first=step_states[-1][0].contiguous() if valid else last
 pooled_ok=bool(valid and layout['available'])
 if pooled_ok:
  rows=step_states[-1][layout['body_state_step_indices']].double().numpy()
  mean=np.mean(rows,axis=0,dtype=np.float64);pool=torch.from_numpy(mean).to(torch.bfloat16).contiguous()
 else:pool=last
 return {'last':last,'answer_prefill':first,'answer_pool':pool}, {'answer_prefill_available':valid,'answer_pool_available':pooled_ok,'answer_prefill_fallback':None if valid else 'no_completed_strict_answer_after_consumed_tool','answer_pool_fallback':None if pooled_ok else (layout.get('reason') or 'no_completed_strict_answer_after_consumed_tool'),'pool_valid_input_tokens':len(layout.get('body_token_indices',[])) if pooled_ok else 0}
