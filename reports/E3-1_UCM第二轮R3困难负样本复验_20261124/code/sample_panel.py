import sys
import torch
from base import RUN,ASSETS,read,dump,sha,table,rgb,setup,panel,weight,model_path,savept
from policy import policy
from train import sample,finish
from rollout import rollout_one,SYSTEM_PROMPT

@torch.inference_mode()
def prepare():
    from prompt_clip import clip
    from tools import scene_classify_with_conf
    m,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    m=m.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head']);head=head.eval().cuda()
    t=table();out={}
    for c in panel():
        a,p=scene_classify_with_conf(rgb(t,c),model=m,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
        out[c['image']]=[a,float(p)]
    dump(RUN/'records/tools.json',out)

class AfterTool(Exception):pass

@torch.inference_mode()
def run(seed, extra=False):
    rows=panel();sig=weight(seed)
    model,proc=policy(model_path(seed));gen=model.generate;t=table();tools=read(RUN/'records/tools.json')
    name=('extra_' if extra else '')+seed
    out=RUN/'records'/name;out.mkdir(exist_ok=True)
    for c in rows:
        path=out/(c['image']+'.pt')
        if path.is_file():
            r=torch.load(path,weights_only=True,map_location='cpu')
            assert r['adapter']==sig and r['rgb']==c['rgb_sha256'] and r['query']==c['scene_query'];continue
        im=rgb(t,c);q=c['scene_query'];answer=tools[c['image']][0]
        first=[];ctx={};calls=[]
        def generate(**kw):
            if first:
                ctx['prefix']=kw['input_ids'].cpu().clone()
                raise AfterTool
            ans=gen(**kw);first.extend(ans.sequences[0,kw['input_ids'].shape[1]:].tolist())
            return ans
        def scene(_):
            calls.append(answer);return answer
        model.generate=generate
        try:
            h,used,filtered=rollout_one(model,proc,im,q,{'scene_classify':scene},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
            assert used==filtered==0 and not calls
        except AfterTool:
            assert calls==[answer]
        finally:model.generate=gen
        sampling_seed=(2242000 if extra else 1141000)+c['row']
        if ctx:
            torch.manual_seed(sampling_seed)
            prefix,records=sample(model,proc,im,q,ctx)
            states,third=finish(model,proc,prefix,records,answer)
            ids=[v[0] for v in records]
        else:
            states=[h]*8;ids=[[] for _ in states];third=[[] for _ in states]
        hs=torch.stack(states).cpu()
        assert hs.shape==(8,3584) and hs.dtype==torch.bfloat16 and torch.isfinite(hs).all()
        savept(path,dict(image=c['image'],rgb=c['rgb_sha256'],query=q,adapter=sig,seed=sampling_seed,
                first=first,second=ids,third=third,tool_calls=calls,post_tool=bool(ctx),hidden=hs))
        print(seed,c['image'],flush=True)
    dump(out/'complete.json',dict(adapter=sig,files={c['image']:sha(out/(c['image']+'.pt')) for c in rows}))
if __name__=='__main__':
    setup()
    if sys.argv[1]=='prepare':prepare()
    else:run(sys.argv[1],'--extra' in sys.argv)
