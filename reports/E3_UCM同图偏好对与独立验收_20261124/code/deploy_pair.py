"""One free, greedy deployment trajectory per image and checkpoint."""
import sys,torch
from env_pair import ROOT as RUN,HIST,PREVIOUS,REBUILD,model_path,weight,read,dump,sha,table,rgb,setup,savept
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT
from shared import ASSETS
VAL=REBUILD/'paired_views_20261120/inputs'
def datasets():
    vc=read(VAL/'validation.json')
    assert sha(VAL/'validation.json')==read(PREVIOUS/'initial.json')['val']
    cf=read(HIST/'inputs/panel.json');assert sha(HIST/'inputs/panel.json')==read(HIST/'inputs/lock.json')['panel']
    ans=read(PREVIOUS/'tools.json')
    assert sha(PREVIOUS/'tools.json')==read(PREVIOUS/'prepare.json')['tools']
    return {'val':(table('val'),vc,{c['image']:ans[c['rgb_sha256']][0] for c in vc},'orig_query'),
            'fresh':(table('train'),cf,{k:v[0] for k,v in read(HIST/'records/tools.json').items()},'scene_query')}
@torch.inference_mode()
def main(seed,reward):
    setup();ds=datasets();tag=f'{seed}_{reward}'
    path=model_path(seed) if reward=='start' else RUN/'pilot'/tag
    sig=weight(seed) if reward=='start' else read(path/'complete.json')['final']
    assert sha(path/'adapter/adapter_model.safetensors')==sig
    m,proc=policy(path);gen=m.generate
    for name,(t,rows,answers,qkey) in ds.items():
        if name=='val' and reward=='start':continue
        out=RUN/'pilot/greedy'/tag/name;out.mkdir(parents=True,exist_ok=True)
        for c in rows:
            nm=c['image'];f=out/(nm+'.pt');rounds=[];calls=[]
            if f.is_file():
                old=torch.load(f,weights_only=True);assert old['adapter']==sig and old['rgb']==c['rgb_sha256'];continue
            def record(**kw):
                resp=gen(**kw);rounds.append(resp.sequences[0,kw['input_ids'].shape[1]:].tolist());return resp
            def scene(_):calls.append(answers[nm]);return calls[-1]
            m.generate=record
            try:
                h,used,filtered=rollout_one(m,proc,rgb(t,c),c[qkey],{'scene_classify':scene},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
            finally:m.generate=gen
            assert h.shape==(3584,) and torch.isfinite(h).all() and used==len(calls) and filtered==0
            savept(f,dict(image=nm,rgb=c['rgb_sha256'],query=c[qkey],adapter=sig,prior=h.cpu(),rounds=rounds,tools=calls))
        dump(out/'complete.json',dict(adapter=sig,files={c['image']:sha(out/(c['image']+'.pt')) for c in rows}))
        print(tag,name,len(rows),flush=True)
if __name__=='__main__':main(*sys.argv[1:])
