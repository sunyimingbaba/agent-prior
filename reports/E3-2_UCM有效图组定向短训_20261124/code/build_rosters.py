from collections import defaultdict,Counter
import torch
from common import ROOT,OLD,read,dump,pool,key,sha
def main():
    rows=pool();proof={}
    for seed in ('a','b'):
        eligible=[];available=[];audit=[]
        for sub in ('','confirm'):
            base=OLD/sub;extra=ROOT/'screen' if sub else OLD
            bank=torch.load(base/'records/text.pt',weights_only=True);h=bank['hard'].double()
            assert sha(base/'records/text.pt')==read(base/'records/read_complete.json')['text']
            for c in [v for v in rows.values() if v['source']==sub]:
                nm=c['image'];a,b=[bank['maps'][nm][k] for k in 'AB'];scores=[]
                for d,tag in [(base,seed),(extra,'extra_'+seed)]:
                    folder=d/'records'/(tag+'_readouts');p=folder/(nm+'.pt');sig=read(folder/'complete.json')
                    assert sig['files'][nm]==sha(p)
                    z=torch.load(p,weights_only=True);v=z['visual'].double();post=z['post_tool']
                    sim=torch.stack([h@x for x in v]);scores.extend(zip((sim[:,a[0]]-sim[:,a[1]]).tolist(),(sim[:,b[0]]-sim[:,b[1]]).tolist()))
                good=sum(x>0 and y>0 for x,y in scores);bad=sum(x<0 and y<0 for x,y in scores)
                if post:available.append(nm)
                if post and good and bad:eligible.append(nm)
                audit.append(dict(image=nm,post=post,both_correct=good,both_wrong=bad))
        ordered=sorted(eligible,key=lambda n:key(seed+'|select|'+n))
        assert ordered
        roster=[]
        for cycle in range(32):
            roster+=sorted(ordered,key=lambda n:key(f'{seed}|{cycle}|{n}'))
            if len(roster)>=32:break
        roster=roster[:32]
        def stratum(n):return tuple(rows[n][k] for k in ('scene_id','attribute','side'))
        by=defaultdict(list)
        for n in available:by[stratum(n)].append(n)
        mapping={};seen=Counter()
        for n in sorted(set(roster),key=lambda n:key(seed+'|map|'+n)):
            k=stratum(n);options=sorted(by[k],key=lambda v:key(seed+'|control|'+v))
            mapping[n]=options[seen[k]];seen[k]+=1
        assert len(set(mapping.values()))==len(mapping)
        match=[mapping[n] for n in roster];assert all(stratum(a)==stratum(b) for a,b in zip(roster,match))
        proof[seed]=dict(available=len(available),eligible=eligible,selected=roster,regular=match,selection_counts=audit)
    dump(ROOT/'inputs/rosters.json',proof)
if __name__=='__main__':main()
