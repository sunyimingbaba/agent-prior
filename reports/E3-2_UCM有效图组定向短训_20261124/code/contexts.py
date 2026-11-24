import sys,torch
from common import ROOT,OLD,read,sha,table,rgb,setup,pool,model_path,savept
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT
class Captured(Exception):pass
@torch.inference_mode()
def main(seed):
    setup();r=read(ROOT/'inputs/rosters.json')[seed];ps=pool();t=table();m,proc=policy(model_path(seed));gen=m.generate;out={}
    for nm in sorted(set(r['regular']+r['selected'])):
        c=ps[nm];folder=OLD/c['source'];arch=torch.load(folder/'records'/seed/(nm+'.pt'),weights_only=True)
        ans=read(folder/'records/tools.json')[nm][0];ids=[]
        def capture(**kw):
            if ids:
                out[nm]=dict(first=ids,prefix=kw['input_ids'].cpu().clone(),answer=ans);raise Captured()
            z=gen(**kw);ids.extend(z.sequences[0,kw['input_ids'].shape[1]:].tolist());return z
        m.generate=capture
        try:rollout_one(m,proc,rgb(t,c),c['scene_query'],{'scene_classify':lambda _:ans},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
        except Captured:pass
        finally:m.generate=gen
        assert nm in out and ids==arch['first'] and arch['tool_calls']==[ans]
    savept(ROOT/'records'/(seed+'_contexts.pt'),out)
if __name__=='__main__':main(sys.argv[1])
