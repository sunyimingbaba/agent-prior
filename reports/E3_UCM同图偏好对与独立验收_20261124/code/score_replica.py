"""Second fixed-policy panel and stratified image-pair replication of the single-prior result."""
import csv
from collections import defaultdict
import numpy as np
from candidate_guard import ROOT, load, read, dump, boot
from score_blind import fixed
from multi_caption import digest
from recheck_reward import rr

def main():
    base=ROOT/'reward_confirm'
    panel=read(base/'inputs/panel.json')
    bank=load(base/'records/text.pt')
    hh=[digest(x) for x in bank['native_tokens'].tolist()]
    odd=np.array([int(x,16)%2==1 for x in hh])
    nomination=read(ROOT/'inputs/nomination_bounded.json')
    anchor=read(ROOT/'records/environment.json')['starting']
    priors={}; rows=[]; levels={}
    for seed in 'ab':
        for arm in ('start','bounded'):
            z=load(ROOT/'pilot/vectors'/(seed+'_'+arm+'_reward_anchor.pt'))
            sig=anchor[seed] if arm=='start' else nomination['adapters'][seed]
            assert z['adapter']==sig and z['images']==[r['image'] for r in panel]
            hard=fixed(z['features'],bank['hard'])
            nat=fixed(z['features'],bank['native'])
            vals=[]
            for c,s,h in zip(panel,nat,hard):
                ix=bank['maps'][c['image']]
                p=ix['native'];alts=sorted(set(p)-{p[0]},key=hh.__getitem__)
                held=alts[1:] if len(alts)>1 else alts;neg=odd.copy();neg[p]=False
                votes=[float(h[ix[k][0]]>h[ix[k][1]]) for k in 'CDE']
                reciprocal=rr(np.array([s[held].max()]),s[neg][None]).item()
                vals.append([np.mean(votes),reciprocal,*votes])
                rows.append(dict(seed=seed,arm=arm,image=c['image'],pair=c['pair'],CDE=float(np.mean(votes)),rr=reciprocal,C=votes[0],D=votes[1],E=votes[2]))
            priors[(seed,arm)]=np.array(vals)
            levels[seed+'_'+arm]=dict(zip(['CDE','rr','C','D','E'],np.mean(vals,axis=0)))
    baseline=np.mean([priors[(s,'start')][:,:2] for s in 'ab'],axis=(0,1))
    assert np.allclose(baseline,read(ROOT/'results/single_vs_candidates.json')['single_levels'],atol=1e-12)
    z=np.array([priors[(s,'bounded')]-priors[(s,'start')] for s in 'ab'])
    groups=[c['pair'] for c in panel]
    comparison={k:dict(**boot(z[:,:,i].mean(0),groups),seeds=z[:,:,i].mean(1).tolist()) for i,k in enumerate(['CDE','rr','C','D','E'])}
    old=read(ROOT/'confirm/inputs/panel.json')
    with (ROOT/'results/blind.csv').open(encoding='utf-8-sig') as fh:
        observed={(r['tag'],r['image']):float(r['score']) for r in csv.DictReader(fh) if r['metric']=='CDE'}
    first=np.array([[observed[(s+'_bounded',r['image'])]-observed[(s+'_start',r['image'])] for r in old] for s in 'ab'])
    assert np.allclose(100*first.mean(axis=1),read(ROOT/'results/blind.json')['difference_pp']['CDE']['seeds'])
    assert not ({r['rgb_sha256'] for r in panel}&{r['rgb_sha256'] for r in old})
    rng=np.random.default_rng(20261124);total=np.zeros(20000)
    for values,details in [(first,old),(z[:,:,0],panel)]:
        pairs=defaultdict(list)
        for i,r in enumerate(details):pairs[r['pair']].append(i)
        assert all(len(v)==2 for v in pairs.values())
        means=np.array([values[:,members].mean() for members in pairs.values()])
        draws=rng.integers(len(pairs),size=(20000,len(pairs)))
        total+=2*means[draws].sum(axis=1)
    joined=np.concatenate((first,z[:,:,0]),axis=1)
    lo,hi=np.quantile(total/joined.shape[1],[.025,.975])
    pooled=dict(mean=float(joined.mean()),lo=lo,hi=hi,seeds=joined.mean(axis=1).tolist(),pairs=(len(panel)+len(old))//2)
    flips={s:{'gained':int((z[i,:,2:]>0).sum()),'lost':int((z[i,:,2:]<0).sum())} for i,s in enumerate('ab')}
    dump(ROOT/'results/replica.json',dict(levels=levels,second_differences=comparison,flips=flips,stratified_detail_pooled=pooled))
    dump(ROOT/'results/replica_rows.json',rows)
if __name__=='__main__':main()
