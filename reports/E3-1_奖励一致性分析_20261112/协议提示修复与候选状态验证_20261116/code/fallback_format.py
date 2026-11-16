"""Conditional fixed30-update format repair; original checkpoints remain immutable."""
import hashlib
import io
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from milestone_common import ROOT,PLAN,ASSETS,seed,lora_hash
from prompt_design import messages
from supervised_format import target_tokens,response_loss
from stage_io import read,dump,sha

def train(model,processor,tables,system):
 d=read(ROOT/'protocol_plan.json');cfg=d['fallback_training'];out=ROOT/'training';out.mkdir(exist_ok=False);seed(cfg['seed']);initial=lora_hash(model)
 assert initial==d['policy']['lora_hash']
 for n,p in model.named_parameters():p.requires_grad_('lora_' in n)
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 params=[p for p in model.parameters() if p.requires_grad];assert all(p.dtype==torch.float32 for p in params)
 sources=read(ROOT/'inputs/teacher_sources.json');assert len(sources)==15;assert not set(x['image'] for x in sources)&{x['image'] for x in read(ROOT/'inputs/cohort.json')}
 table=tables['train'];byname={table['path'][i].as_py():i for i in range(table.num_rows)};tools=read(ROOT/'inputs/sft_actual_teacher_tools.json');eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
 examples=[];images={}
 for source in sources:
  name=source['image'];row=byname[name];image=Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB');images[name]=image
  for prompt in d['prompt_candidates_priority']:
   initial_messages=messages(prompt,{'image':name},system)
   for kind in ['call','answer']:
    used=initial_messages if kind=='call' else initial_messages+[{'role':'assistant','content':'<tool_call>scene_classify</tool_call>'},{'role':'user','content':'<result>'+tools[name]['text']+'</result>'}]
    text=processor.apply_chat_template(used,tokenize=False,add_generation_prompt=True);inputs=processor(text=[text],images=[image],return_tensors='pt');target=target_tokens(processor.tokenizer,kind,source['source_body'] if kind=='answer' else '')
    examples.append({'id':name+'/'+prompt+'/'+kind,'image':name,'prompt':prompt,**target,'prompt_text':text,'prompt_token_ids':inputs.input_ids[0].tolist(),'prompt_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'message_roles':[m['role'] for m in used]})
 dump(out/'teacher_examples.json',examples);group={s['image']:[x for x in examples if x['image']==s['image']] for s in sources}
 optimizer=torch.optim.AdamW(params,lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
 random=np.random.default_rng(cfg['seed']);names=[s['image'] for s in sources];order=[names[i] for epoch in range(2) for i in random.permutation(15)];dump(out/'order.json',order)
 def save(label,step):
  folder=out/label/'checkpoint';folder.mkdir(parents=True,exist_ok=False);model.save_pretrained(str(folder),safe_serialization=True);processor.save_pretrained(str(folder));torch.save(optimizer.state_dict(),folder/'optimizer.pt');torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},folder/'torch_rng.pt')
  dump(folder/'training_state.json',{'optimizer_updates':step,'policy_hash':lora_hash(model),'initial_policy_hash':initial,'order_sha256':sha(out/'order.json')});dump(out/(label+'_manifest.json'),{'files':{str(p.relative_to(folder)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in folder.rglob('*') if p.is_file()},'optimizer_updates':step})
 for step,name in enumerate(order,1):
  model.train();optimizer.zero_grad(set_to_none=True);den=sum(x['supervised_tokens'] for x in group[name]);rows=[]
  for ex in group[name]:
   loss,stats=response_loss(model,processor,images[name],ex,eos);assert torch.isfinite(loss);(loss/den).backward();rows.append({'id':ex['id'],'loss_sum':float(loss.detach()),'stats':stats})
  norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']));assert math.isfinite(norm) and norm>0 and all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
  assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n);before=lora_hash(model);optimizer.step();after=lora_hash(model);assert after!=before
  rec={'step':step,'image':name,'active_tokens':den,'loss':sum(x['loss_sum'] for x in rows)/den,'gradient_norm':norm,'policy_hash_before':before,'policy_hash_after':after,'examples':rows}
  with (out/'updates.jsonl').open('a') as f:f.write(__import__('json').dumps(rec)+'\n');f.flush()
  print('PROTOCOL_FORMAT_UPDATE',step,rec['loss'],flush=True)
  if step==15:save('intermediate_15',step)
 save('final',30);final=lora_hash(model);assert final!=initial
 model.gradient_checkpointing_disable();model.eval()
 for p in model.parameters():p.requires_grad_(False)
 result={'ok':True,'optimizer_updates':30,'teacher_images':15,'prompt_styles':3,'assistant_targets':90,'initial_policy_hash':initial,'final_policy_hash':final,'checkpoint':'training/final/checkpoint','selection':'fixed final30, train protocol gate only'};dump(out/'result.json',result);return result
