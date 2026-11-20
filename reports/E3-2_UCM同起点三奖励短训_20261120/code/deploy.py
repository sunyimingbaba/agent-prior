"""Regenerate native priors after loading each saved adapter, without using readout scores or labels."""
import hashlib,sys,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from shared import ROOT,BASE,ASSETS,read,dump,sha,table,rgb,setup
from rollout import load_mllm,rollout_one,SYSTEM_PROMPT
from tools import scene_classify_with_conf
from prompt_clip import clip

def run(arm,part):
    assert arm in ('base','old','caption','hard') and part in ('native','generic','detail')
    setup()
    adapter=ASSETS/'ucm_main_lora' if arm=='base' else ROOT/('policy_'+arm)/'adapter'
    modelhash=sha(adapter/'adapter_model.safetensors')
    if arm!='base':assert modelhash==read(adapter.parent/'update.json')['adapter_hash']
    for key,expected in read(ROOT/'inputs/eval_lock.json').items():assert sha(ROOT/key)==expected
    out=ROOT/'eval'/arm/part;out.mkdir(parents=True,exist_ok=True)
    model,proc=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(adapter))
    active=hashlib.sha256()
    for name,param in model.named_parameters():
        if '.lora_' in name:active.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    digest=active.hexdigest()
    if arm!='base':assert digest==read(adapter.parent/'update.json')['final_params']
    dump(out/'loaded_weights.json',dict(file_hash=modelhash,loaded_params=digest))
    tool,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    tool=tool.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']));head.load_state_dict(cp['head']);head=head.eval().cuda()
    t=table('val');cases=read(ROOT/'inputs'/('detail.json' if part=='detail' else 'validation.json'))
    original=model.generate;rounds=[]
    def capture(*args,**kwargs):
        obj=original(*args,**kwargs)
        ids=obj.sequences[0,kwargs['input_ids'].shape[-1]:].cpu().tolist()
        rounds.append(dict(ids=ids,text=proc.decode(ids,skip_special_tokens=True)))
        return obj
    model.generate=capture
    n=0
    for r in cases:
        stem=Path(r['image']).stem
        base=rgb(t,r)
        query=r['orig_query'] if part=='native' else (r['scene_query'] if part=='detail' else 'An aerial photograph.')
        modes=['upright','rotated'] if part=='detail' else ['upright']
        for mode in modes:
            im=base if mode=='upright' else base.transpose(Image.Transpose.ROTATE_180)
            seeds=list(range(-1,4)) if part=='detail' else [-1]
            file=out/(stem+'_'+mode+'.pt');info=file.with_suffix('.json')
            if info.exists():
                record=read(info)
                assert record['model']==modelhash and record['query']==query and record['sha256']==sha(file)
                n+=len(seeds);continue
            answer=None;vectors=[];traces=[]
            def scene(img):
                nonlocal answer
                if answer is None:answer=scene_classify_with_conf(img,model=tool,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
                text,conf=answer;events.append(dict(text=text,confidence=float(conf)));return text
            for k in seeds:
                seed=394000+r['row']*10+k+1
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
                rounds.clear();events=[]
                with torch.inference_mode():
                    h,used,filtered=rollout_one(model,proc,im,query,{'scene_classify':scene},3,64,'cuda',temperature=float(k>=0),system_prompt=SYSTEM_PROMPT)
                assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all() and used==len(events) and filtered==0
                vectors.append(h)
                traces.append(dict(k=k,seed=seed,rounds=rounds.copy(),events=events,used=used,hidden=hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()))
            tmp=file.with_suffix('.part');torch.save(torch.stack(vectors),tmp);tmp.replace(file)
            dump(info,dict(model=modelhash,image=r['image'],row=r['row'],rgb_sha256=r['rgb_sha256'],view=mode,
                 rendered_sha256=hashlib.sha256(im.tobytes()).hexdigest(),query=query,traces=traces,sha256=sha(file)))
            n+=len(vectors)
    dump(out/'complete.json',dict(arm=arm,part=part,images=len(cases),trajectories=n,model_sha256=modelhash))
    print('DONE',arm,part,n,flush=True)
if __name__=='__main__':
    from run_once import main as synchronized
    synchronized(*sys.argv[1:])
