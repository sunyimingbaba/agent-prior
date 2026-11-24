import sys
import torch
from safetensors.torch import load_file
from shared import ROOT,ASSETS,rgb,sha,setup,dump,read
from data import DOMAINS,PREVIOUS,STATES,dataset
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT

@torch.inference_mode()
def tools_cache():
    from prompt_clip import clip
    from tools import scene_classify_with_conf
    m,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    m=m.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head'])
    head=head.eval().cuda()
    result={}
    for dn in DOMAINS:
        table,cases=dataset(dn)
        for c in cases:
            a,p=scene_classify_with_conf(rgb(table,c),model=m,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
            result[c['rgb_sha256']]=[a,float(p)]
    dump(ROOT/'tools.json',result)
    baselines={}
    for sub in ('','repeat'):
        path=PREVIOUS/sub/'R0/adapter/adapter_model.safetensors'
        weights=load_file(str(path),device='cpu')
        b=[v for k,v in weights.items() if '.lora_B.' in k]
        assert b and all(v.count_nonzero()==0 for v in b) and all('.lora_A.' in k or '.lora_B.' in k for k in weights)
        baselines[sub or 'a']=sha(path)
    dump(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,torch=str(torch.__version__),
         r0_zero_B=baselines,scene_head=sha(ASSETS/'ucm_scene_cls.pt'),tools=sha(ROOT/'tools.json')))

@torch.inference_mode()
def infer(state):
    sub,arm=STATES[state]; path=PREVIOUS/sub/arm
    weight=sha(path/'adapter/adapter_model.safetensors')
    expected=read(PREVIOUS/sub/'initial.json')['adapter'] if arm=='R0' else read(path/'complete.json')['final']
    assert weight==expected
    m,proc=policy(path)
    cache=read(ROOT/'tools.json')
    assert sha(ROOT/'tools.json')==read(ROOT/'environment.json')['tools']
    original=m.generate
    def record(**kw):
        response=original(**kw)
        ids=response.sequences[0,kw['input_ids'].shape[1]:].tolist()
        rounds.append(dict(ids=ids,text=proc.decode(ids,skip_special_tokens=True)))
        return response
    m.generate=record
    for dn in DOMAINS:
        table,cases=dataset(dn);folder=ROOT/'outputs'/state/dn;folder.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(cases):
            f=folder/f"{c['row']:04d}.pt"
            if f.exists():
                old=torch.load(f,map_location='cpu',weights_only=True)
                assert (old['adapter'],old['rgb'],old['query'])==(weight,c['rgb_sha256'],c['orig_query'])
                continue
            rounds=[]; events=[]
            def scene(im):
                a,p=cache[c['rgb_sha256']]
                events.append((a,p));return a
            h,used,filtered=rollout_one(m,proc,rgb(table,c),c['orig_query'],{'scene_classify':scene},3,64,
                                        'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
            assert h.shape==(3584,) and h.dtype==torch.bfloat16 and h.isfinite().all() and used==len(events) and filtered==0
            tmp=f.with_suffix('.part')
            torch.save(dict(image=c['image'],row=c['row'],rgb=c['rgb_sha256'],query=c['orig_query'],adapter=weight,
                            rounds=rounds,tools=events,prior=h.cpu()),tmp)
            tmp.replace(f)
            if (i+1)%50==0:print(state,dn,i+1,flush=True)
        dump(folder/'complete.json',dict(adapter=weight,rows=[sha(folder/f"{c['row']:04d}.pt") for c in cases]))
if __name__=='__main__':
    setup()
    if sys.argv[1]=='prepare':tools_cache()
    else:infer(sys.argv[1])
