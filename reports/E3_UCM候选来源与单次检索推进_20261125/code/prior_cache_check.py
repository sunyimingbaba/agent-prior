"""Check exact hidden-state keys against independently archived CUP embeddings and proxy ranks."""
from collections import Counter
import sys
import torch
from remote_common import ROOT,BASE,read,sha,dump
STORE=BASE/'paired_answer_preference_20261124/guard/records'
def old_states(seed,c,sig):
    cache={}
    for k in (seed,'extra_'+seed):
        p=STORE/k/(c['image']+'.pt');z=torch.load(p,weights_only=True);q=torch.load(STORE/(k+'_readouts')/p.name,weights_only=True)
        assert z['adapter']==sig and z['rgb']==c['rgb_sha256'] and q['source']==sha(p)
        assert z['image']==q['image']==c['image'] and z['query']==c['scene_query']
        for h,v,r in zip(z['hidden'],q['visual'],q['proxy_rank']):
            key=h.float().numpy().tobytes()
            if key in cache:assert torch.equal(v,cache[key][0]) and r==cache[key][1]
            cache[key]=(v,r)
    return cache
def check():
    torch.set_num_threads(1);counts=Counter();maxdiff=0.
    for s in 'ab':
        for c in read(ROOT/'inputs/third.json'):
            base=ROOT/'records/prefix/third'/s;p=base/'features'/(c['image']+'.pt')
            if not p.exists():continue
            f=torch.load(p,weights_only=True);raw=torch.load(base/'raw'/p.name,weights_only=True)
            cache=old_states(s,c,f['adapter']);checked=set()
            for k,z in raw['modes'].items():
                for h,v,r in zip(z['hidden'],f['modes'][k]['visual'],f['modes'][k]['ranks']):
                    key=h.float().numpy().tobytes()
                    if key in checked:continue
                    checked.add(key);counts['distinct']+=1
                    if key in cache:
                        w,rank=cache[key];err=float((w-v).abs().max());maxdiff=max(maxdiff,err)
                        counts['matched']+=1;counts['rank_difference']+=(rank!=r);counts['visual_bits_differ']+=not torch.equal(w,v)
            counts['images']+=1
    out=dict(counts=dict(counts),max_feature_difference=maxdiff)
    dump(ROOT/'results/cache_check.json',out);print(out)
if __name__=='__main__':check()
