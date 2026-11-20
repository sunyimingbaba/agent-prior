"""Inspect native generated tokens and compare a fixed original-caption control."""
from common import ASSETS, ROOT, BASE, QUERY, read, dump, tables, image_for
import hashlib
import torch
from rollout import load_mllm,rollout_one,SYSTEM_PROMPT
from tools import scene_classify_with_conf
from prompt_clip import clip

data=tables()
records=read(ROOT/'inputs/scored_images.json')
model,proc=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(ASSETS/'ucm_main_lora'))
original=model.generate
rounds=[]
def capture(*args,**kwargs):
    out=original(*args,**kwargs)
    ids=out.sequences[0,kwargs['input_ids'].shape[-1]:].cpu().tolist()
    rounds.append(dict(ids=ids,tokens=proc.tokenizer.convert_ids_to_tokens(ids),decoded=proc.decode(ids,skip_special_tokens=True)))
    return out
model.generate=capture
clip_model,preprocess=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
clip_model=clip_model.float().eval().cuda()
cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']));head.load_state_dict(cp['head']);head=head.cuda().eval()
def scene(im):
    t,c=scene_classify_with_conf(im,model=clip_model,preprocess=preprocess,cls_head=head,cls_names=list(cp['classes']))
    return t if c>=0 else None
results=[]
eos=model.generation_config.eos_token_id
eos={eos} if isinstance(eos,int) else set(eos)
# Six existing quantity controls, with train and validation images.
control={'181.tif','984.tif','2088.tif','1937.tif','147.tif','102.tif'}
for x in records:
    queries=[('generic',QUERY)]
    if x['image'] in control:
        queries.append(('original',data[x['split']]['caption'][x['row']].as_py()[0]))
    for mode,q in queries:
        torch.manual_seed(0);rounds.clear()
        h,u,_=rollout_one(model,proc,image_for(x,data),q,{'scene_classify':scene},3,64,'cuda',
                          temperature=0.,system_prompt=SYSTEM_PROMPT)
        if mode=='generic':
            prior=ROOT/'priors/AGSE'/(x['image'].split('.')[0]+'.pt')
        else:
            prior=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool'/('ucm_caption_karpathy_label_'+x['split'])/(str(x['row'])+'.pt')
        ref=torch.load(prior,map_location='cpu',weights_only=True).reshape_as(h)
        eq=torch.equal(h,ref)
        if mode=='generic':assert eq
        results.append(dict(image=x['image'],split=x['split'],mode=mode,query=q,rounds=list(rounds),tools=int(u),
                            first_output_is_single_EOS=len(rounds[0]['ids'])==1 and rounds[0]['ids'][0] in eos,
                            same_hidden=eq,hidden_cos=float(torch.nn.functional.cosine_similarity(h.float().reshape(1,-1),ref.float().reshape(1,-1)))))
dump(ROOT/'native_token_control.json',{'configured_eos_ids':sorted(eos),'records':results})
print('verified',len(results))
