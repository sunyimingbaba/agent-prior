"""Sample the frozen original AGSE deployment rollout, including genuine tool responses."""
from pathlib import Path
import hashlib,random
import numpy as np
import torch
from shared import ROOT,BASE,ASSETS,read,dump,sha,table,rgb,setup,locked
from rollout import load_mllm,rollout_one,SYSTEM_PROMPT
from tools import scene_classify_with_conf
from prompt_clip import clip

def main(cases=None, object_views=False, output=None):
    setup()
    if cases is None:cases=read(ROOT/"inputs/selected.json")
    output=ROOT if output is None else output
    tables={s:table(s) for s in ("train","val")}
    model,proc=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(ASSETS/'ucm_main_lora'))
    tool,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    tool=tool.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head'],strict=True)
    head=head.eval().cuda()
    original=model.generate
    rounds=[]
    def capture(*args,**kwargs):
        assert kwargs['do_sample']==(k>=0)
        if k>=0:assert kwargs['temperature']==1.
        out=original(*args,**kwargs)
        ids=out.sequences[0,kwargs['input_ids'].shape[-1]:].cpu().tolist()
        rounds.append(dict(ids=ids,decoded=proc.decode(ids,skip_special_tokens=True)))
        return out
    model.generate=capture
    for r in cases:
        base_im=rgb(tables[r['split']],r)
        im=base_im
        answer=None
        def scene(img):
            nonlocal answer
            assert img is im
            if answer is None:
                answer=scene_classify_with_conf(img,model=tool,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
            text,conf=answer
            events.append(dict(text=text,confidence=float(conf)))
            return text
        modes=[('scene','scene_query'),('original','orig_query')]
        if r['split']=='val' and r['contrast'] in ('hv','diag','lr','tb','nose'):modes.append(('rotated','scene_query'))
        if object_views:modes=[('scene','scene_query'),('rotated','scene_query')]
        for mode,key in modes:
            if mode=='rotated':
                from PIL import Image
                transform=Image.Transpose.ROTATE_90 if r['contrast'] in ('hv','diag') else Image.Transpose.ROTATE_180
                im=base_im.transpose(transform);answer=None
            dst=output/'samples'/mode/Path(r['image']).stem
            dst.mkdir(parents=True,exist_ok=True)
            for k in range(-1,8):
                name='greedy' if k<0 else f'sample{k}'
                info=dst/(name+'.json');tensor=dst/(name+'.pt')
                seed=20261120+r['row']*40+(10 if mode=='original' else 0)+k+1
                if info.exists():
                    prev=read(info)
                    assert prev['seed']==seed and prev['sha256']==sha(tensor) and prev['query']==r[key]
                    continue
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
                rounds.clear();events=[]
                with torch.inference_mode():
                    h,used,filtered=rollout_one(model,proc,im,r[key],{'scene_classify':scene},3,64,'cuda',
                                                temperature=float(k>=0),system_prompt=SYSTEM_PROMPT)
                assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
                history=None
                if mode=='original' and k==-1:
                    file=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool'/('ucm_caption_karpathy_label_'+r['split'])/(str(r['row'])+'.pt')
                    old=torch.load(file,map_location='cpu',weights_only=True).reshape_as(h)
                    history=dict(archive_sha=sha(file),bit_equal=torch.equal(old,h))
                part=tensor.with_suffix('.part')
                torch.save(h,part);part.replace(tensor)
                dump(info,dict(image=r['image'],row=r['row'],rgb_sha256=r['rgb_sha256'],mode=mode,k=k,query=r[key],seed=seed,
                               rounds=rounds,events=events,used=used,filtered=filtered,history=history,sha256=sha(tensor),
                               vector_sha256=hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
        print('generated',r['image'],flush=True)
    print('DONE',len(cases),'source photos',flush=True)
if __name__=='__main__':main()
