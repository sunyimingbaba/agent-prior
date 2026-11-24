import hashlib,sys
from collections import defaultdict
import torch
from base import RUN,read,dump,sha,panel,table,rgb,setup,model_path,savept
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT
def key(s):
    return hashlib.sha256(('r3-pilot-20261124-'+s).encode()).hexdigest()
class Captured(Exception):pass
@torch.inference_mode()
def main(seed):
    setup();dest=RUN/'pilot';dest.mkdir(exist_ok=True);p=panel();t=table();groups=defaultdict(list)
    for c in p:groups[c['pair']].append(c)
    def good(pair):
        return all(torch.load(RUN/'records'/a/(c['image']+'.pt'),weights_only=True)['post_tool'] for a in ('a','b') for c in groups[pair])
    eligible=sorted((g for g in groups if good(g)),key=key)
    chosen=[];scenes=set()
    for g in eligible:
        sc=groups[g][0]['scene_id']
        if sc not in scenes:scenes.add(sc);chosen.append(g)
    chosen=(chosen+[g for g in eligible if g not in chosen])[:16];assert len(chosen)==16
    cases=sorted([r for g in chosen for r in groups[g]],key=lambda r:key(r['image']))
    if seed=='a':dump(dest/'train.json',cases)
    else:assert cases==read(dest/'train.json')
    m,proc=policy(model_path(seed));gen=m.generate;answers=read(RUN/'records/tools.json');contexts={}
    for c in cases:
        nm=c['image'];prev=torch.load(RUN/'records'/seed/(nm+'.pt'),weights_only=True);ids=[]
        def capture(**kw):
            if ids:
                contexts[nm]=dict(first=ids,prefix=kw['input_ids'].cpu().clone(),answer=answers[nm][0]);raise Captured()
            out=gen(**kw);ids.extend(out.sequences[0,kw['input_ids'].shape[1]:].tolist());return out
        m.generate=capture
        try:rollout_one(m,proc,rgb(t,c),c['scene_query'],{'scene_classify':lambda _:answers[nm][0]},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
        except Captured:pass
        finally:m.generate=gen
        assert nm in contexts and ids==prev['first'] and prev['tool_calls']==[answers[nm][0]]
    savept(dest/(seed+'_contexts.pt'),contexts)
if __name__=='__main__':main(sys.argv[1])
