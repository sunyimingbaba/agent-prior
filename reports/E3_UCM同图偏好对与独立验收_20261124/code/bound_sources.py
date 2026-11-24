import sys,torch
from env_pair import ROOT,OLD,setup,rows,table,rgb,read,model_path,weight,savept
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT
@torch.inference_mode()
def main(seed):
    setup();m,proc=policy(model_path(seed));out={};t=table();ans={}
    for sub in ('','confirm'):ans.update({n:a[0] for n,a in read(OLD/sub/'records/tools.json').items()})
    for nm,c in rows().items():
        out[nm]={}
        for field in ('scene_query','orig_query'):
            calls=[];h,n,f=rollout_one(m,proc,rgb(t,c),c[field],{'scene_classify':lambda _:calls.append(ans[nm]) or ans[nm]},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
            assert f==0 and n==len(calls);out[nm][field]=h.cpu()
    savept(ROOT/'records'/(seed+'_bound_source.pt'),dict(adapter=weight(seed),rows=out))
if __name__=='__main__':main(sys.argv[1])
