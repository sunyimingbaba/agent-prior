import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
from milestone_common import ROOT,PLAN,ASSETS,setup,seed,dump,sha,case_images,load_policy,real_tools,lora_hash
from checkpoint_io import restore_adapter_exact
from live_rollout_legacy import messages_for
from supervised_format import target_tokens,response_loss

torch=setup();seed(PLAN['training']['seed']);start=time.monotonic()
from agent.rollout import SYSTEM_PROMPT
out=ROOT/'training';out.mkdir(exist_ok=False)
cases=json.loads((ROOT/'inputs/cases.json').read_text());by={x['image']:x for x in cases}
sources=json.loads((ROOT/'inputs/teacher_sources.json').read_text());assert len(sources)==15
images=case_images([by[x['image']] for x in sources]);tools=real_tools()
actual={name:tools['scene_classify'](x['image']) for name,x in images.items()}
old_actual=json.loads((ROOT/'baseline_evidence/actual_tools.json').read_text())
for name,x in actual.items():assert x['text']==old_actual[name]['text'] and abs(x['confidence']-old_actual[name]['confidence'])<1e-6
model,processor=load_policy(Path(PLAN['policy']['checkpoint']),True)
restore=restore_adapter_exact(model,Path(PLAN['policy']['checkpoint']));initial=lora_hash(model)
assert initial==PLAN['policy']['lora_tensor_hash']
parameters=[p for n,p in model.named_parameters() if p.requires_grad]
assert all(('lora_' in n)==p.requires_grad for n,p in model.named_parameters())
assert all(p.dtype==torch.float32 for p in parameters)
eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos)
snapshot=ROOT/'tokenizer_snapshot';snapshot.mkdir()
(snapshot/'training_tokenizer.json').write_text(processor.tokenizer.backend_tokenizer.to_str(),encoding='utf-8')
assert sha(snapshot/'training_tokenizer.json')==json.loads((ROOT/'tokenization_gate/result.json').read_text())['tokenizer_snapshot_sha256']
for name in ['tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt']:
    p=ASSETS/'models/Qwen2.5-VL-7B-Instruct'/name
    if p.exists():shutil.copy2(p,snapshot/name)
examples=[]
for source in sources:
    image=images[source['image']]['image'];messages=messages_for(PLAN['prompt'],SYSTEM_PROMPT)
    for kind in ['call','answer']:
        used=messages if kind=='call' else messages+[
            {'role':'assistant','content':'<tool_call>scene_classify</tool_call>'},
            {'role':'user','content':'<result>'+actual[source['image']]['text']+'</result>'}]
        rendered=processor.apply_chat_template(used,tokenize=False,add_generation_prompt=True)
        assert rendered.endswith('<|im_start|>assistant\n')
        inputs=processor(text=[rendered],images=[image],return_tensors='pt')
        target=target_tokens(processor.tokenizer,kind,source['source_body'] if kind=='answer' else '')
        assert target['eos_id'] in eos and target['target_ids'][0] not in eos
        examples.append({'id':source['image']+'__'+kind,'image':source['image'],'source_record_id':source['source_record_id'],
            **target,'prompt_text':rendered,'prompt_text_sha256':__import__('hashlib').sha256(rendered.encode()).hexdigest(),
            'prompt_token_ids':inputs.input_ids[0].tolist(),'prompt_tokens':int(inputs.input_ids.shape[1]),
            'message_roles':[x['role'] for x in used],'image_grid_thw':inputs['image_grid_thw'].tolist(),
            'image_merge_size':int(processor.image_processor.merge_size),'eos_token_ids':eos})
dump(out/'teacher_examples.json',examples);dump(out/'actual_teacher_tools.json',actual)
dump(out/'settings.json',{'initial_parameter_hash':initial,'restoration':restore,'training':PLAN['training'],
    'trainable_tensors':len(parameters),'trainable_parameters':sum(p.numel() for p in parameters),
    'base_parameters_frozen':True,'all_adapter_parameters_float32':True,
    'tokenizer_snapshot_sha256':sha(snapshot/'training_tokenizer.json')})
group={name:[x for x in examples if x['image']==name] for name in images}
witness_image=sources[0]['image']

def forward(example,capture=False):
    box={};handle=None
    if capture:
        p=len(example['prompt_token_ids']);n=len(example['target_ids']);mask=torch.tensor(example['loss_mask'],device='cuda')
        def hook(module,args,output):
            box['logits']=output.logits[0,p-1:p+n-1][mask].detach().cpu().contiguous()
        handle=model.register_forward_hook(hook)
    try:value,stats=response_loss(model,processor,images[example['image']]['image'],example,eos)
    finally:
        if handle is not None:handle.remove()
    return value,stats,box.get('logits')

@torch.no_grad()
def teacher_evaluation(label):
    model.eval();rows=[];witness={}
    for example in examples:
        capture=example['image']==witness_image;value,stats,logits=forward(example,capture)
        rows.append({'example_id':example['id'],'loss_sum':float(value),'stats':stats})
        if capture:witness[example['id']]=logits
    torch.save(witness,out/('logit_witness_'+label+'.pt'))
    result={'ok':True,'label':label,'examples':rows,'witness_image':witness_image,
        'active_tokens':sum(x['stats']['active_tokens'] for x in rows),
        'loss_sum':math.fsum(x['loss_sum'] for x in rows),
        'active_argmax_correct':sum(x['stats']['active_argmax_correct'] for x in rows),
        'parameter_hash':lora_hash(model)}
    result['global_mean_nll']=result['loss_sum']/result['active_tokens']
    dump(out/('teacher_evaluation_'+label+'.json'),result)
    return result

before=teacher_evaluation('initial');assert before['parameter_hash']==initial
cfg=PLAN['training'];optimizer=torch.optim.AdamW(parameters,lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
# Seeded real backward witness, followed by zeroing gradients. No optimizer step.
model.train();optimizer.zero_grad(set_to_none=True);warm=[]
denominator=sum(x['supervised_tokens'] for x in group[witness_image])
for example in group[witness_image]:
    value,stats,_=forward(example);(value/denominator).backward();warm.append({'example_id':example['id'],'loss_sum':float(value.detach()),'stats':stats})
grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters,float('inf')))
assert math.isfinite(grad_norm) and grad_norm>0 and all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
assert lora_hash(model)==initial
dump(out/'gradient_witness.json',{'ok':True,'optimizer_updates':0,'image':witness_image,'gradient_norm':grad_norm,
 'parameter_hash_unchanged':initial,'examples':warm,'prompt_and_body_token_losses_excluded':True})
optimizer.zero_grad(set_to_none=True)
names=[x['image'] for x in sources];rng=np.random.default_rng(cfg['seed']);order=[]
for epoch in range(cfg['epochs']):
    order += [{'epoch':epoch+1,'image':names[int(i)]} for i in rng.permutation(len(names))]
assert len(order)==30;dump(out/'order.json',order)
updates=[];previous_hash=initial

def save_checkpoint(label,step,current_hash):
    folder=out/label/'checkpoint';folder.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(str(folder),safe_serialization=True);processor.save_pretrained(str(folder))
    torch.save(optimizer.state_dict(),folder/'optimizer.pt')
    torch.save({'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()},folder/'torch_rng.pt')
    ns=np.random.get_state()
    dump(folder/'python_numpy_rng.json',{'python_random_state':random.getstate(),
        'numpy_random_state':{'kind':ns[0],'keys':ns[1].tolist(),'position':ns[2],'has_gauss':ns[3],'cached_gaussian':ns[4]}})
    dump(folder/'training_state.json',{'optimizer_updates':step,'parameter_hash':current_hash,'initial_parameter_hash':initial,
       'next_order_index':step,'training_order_sha256':sha(out/'order.json'),'plan_sha256':sha(ROOT/'plan.json')})
    files={str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.rglob('*') if p.is_file()}
    dump(out/(label+'_checkpoint_manifest.json'),{'ok':True,'optimizer_updates':step,'parameter_hash':current_hash,'files':files})

for step,item in enumerate(order,1):
    model.train();optimizer.zero_grad(set_to_none=True);rows=[]
    denominator=sum(x['supervised_tokens'] for x in group[item['image']])
    for example in group[item['image']]:
        value,stats,_=forward(example);assert torch.isfinite(value)
        (value/denominator).backward();rows.append({'example_id':example['id'],'loss_sum':float(value.detach()),'stats':stats})
    grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip_norm']))
    assert math.isfinite(grad_norm) and grad_norm>0 and all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
    optimizer.step();current_hash=lora_hash(model)
    assert current_hash!=previous_hash and all(torch.isfinite(p).all() for p in parameters)
    record={'step':step,'epoch':item['epoch'],'image':item['image'],'optimizer_update':True,
       'supervised_tokens':denominator,'loss':math.fsum(x['loss_sum'] for x in rows)/denominator,
       'gradient_norm_before_clip':grad_norm,'parameter_hash_before':previous_hash,'parameter_hash_after':current_hash,
       'examples':rows}
    updates.append(record)
    with (out/'updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    previous_hash=current_hash
    dump(ROOT/'progress.json',{'phase':'training','optimizer_updates':step,'total_updates':30,'loss':record['loss']})
    print('SFT_UPDATE',step,'/30','loss',record['loss'],'grad',grad_norm,flush=True)
    if step==15:save_checkpoint('intermediate_15',step,current_hash)
assert len(updates)==30 and lora_hash(model)!=initial
after=teacher_evaluation('final');final_hash=lora_hash(model);assert after['parameter_hash']==final_hash
save_checkpoint('final',30,final_hash)
dump(out/'result.json',{'ok':True,'optimizer_updates':30,'training_images':15,'assistant_examples':30,'epochs':2,
    'initial_parameter_hash':initial,'final_parameter_hash':final_hash,'trainable_tensors':len(parameters),
    'checkpoint':'training/final/checkpoint','checkpoint_selection':'fixed update 30',
    'initial_teacher_nll':before['global_mean_nll'],'final_teacher_nll':after['global_mean_nll'],
    'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
print('FORMAT_SUPERVISION_TRAINING_COMPLETE',final_hash,flush=True)
