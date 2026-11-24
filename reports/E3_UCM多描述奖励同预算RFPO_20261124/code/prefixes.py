import sys
from collections import defaultdict
import torch
from runtime import ROOT,pool,table,rgb,setup,initial,savept,sha,dump,key
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT

@torch.inference_mode()
def greedy(model,proc,im,query,feedback):
    generate=model.generate
    actions=[];calls=[];prefix=[]
    def record(**kw):
        if len(actions)==1:prefix.append(kw['input_ids'].cpu().clone())
        z=generate(**kw);actions.append(z.sequences[0,kw['input_ids'].shape[1]:].tolist());return z
    model.generate=record
    try:h,n,f=rollout_one(model,proc,im,query,{'scene_classify':lambda _:calls.append(feedback) or feedback},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
    finally:model.generate=generate
    assert n==len(calls) and f==0 and h.shape==(3584,) and torch.isfinite(h).all()
    return h.cpu(),actions,calls,(prefix[0] if prefix else None)

def prepare(seed):
    setup();path=initial(seed);model,proc=policy(path);rows=pool();t=table();out={}
    for nm,c in rows.items():
        h,actions,calls,prefix=greedy(model,proc,rgb(t,c),c['scene_query'],c['feedback'])
        assert (prefix is not None)==bool(calls)
        out[nm]=dict(first=actions[0],prefix=prefix,answer=c['feedback'],prior=h,actions=actions,tools=calls)
    savept(ROOT/'records'/(seed+'_contexts.pt'),dict(rows=out,adapter=sha(path/'adapter/adapter_model.safetensors')))

def roster():
    rows=pool();pairs=defaultdict(list)
    for nm,c in rows.items():pairs[c['pair']].append(nm)
    ctx={s:torch.load(ROOT/'records'/(s+'_contexts.pt'),weights_only=True)['rows'] for s in 'ab'}
    eligible=[];omitted=[]
    for pair,group in sorted(pairs.items(),key=lambda p:key('mc-rfpo-pair',p[0])):
        assert len(group)==2 and {rows[n]['side'] for n in group}=={0,1}
        group=sorted(group,key=lambda n:key('mc-rfpo-side',n))
        (eligible if all(ctx[s][n]['prefix'] is not None for s in 'ab' for n in group) else omitted).extend(group)
    assert eligible
    dump(ROOT/'inputs/roster.json',dict(order=eligible,omitted=omitted,contexts={s:sha(ROOT/'records'/(s+'_contexts.pt')) for s in 'ab'},initial={s:sha(initial(s)/'adapter/adapter_model.safetensors') for s in 'ab'},protocol=sha(ROOT/'实验口径.md')))
if __name__=='__main__':
    if sys.argv[1]=='roster':roster()
    else:prepare(sys.argv[1])
