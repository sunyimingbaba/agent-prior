"""Reconstruct the two-image choice from exact leave-one-caption-out winner probabilities."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parent
SOURCES={'C98':REPORTS/'E3_双收益回答对的偏好训练_20261127/validation98/fresh98',
         'D114':REPORTS/'E3_困难奖励跨图跨措辞校准_20261127/followup_pool'}
ARMS=('r2','strict','bidirectional','self')
def read(p):return json.loads(p.read_bytes())
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def base_reward(items):
    margin=np.array([q['margin'] for q in items])
    score=np.sort([1/(1+q['rank']) for q in items])
    out=np.empty(len(items))
    offset=0
    for x in np.unique(margin):
        eq=margin==x; out[eq]=score[offset:offset+eq.sum()].mean(); offset+=eq.sum()
    return out

def protected_winners(r2,truth,guard):
    top=np.flatnonzero(r2==max(r2));weakest=truth.min(axis=1);hits=(truth>0).sum(axis=1)
    retained=[i for i in range(len(r2)) if r2[i]<r2[top[0]] and guard[i] and
              len(top)*hits[i]>hits[top].sum() and weakest[i]>max(weakest[top])]
    if not retained:return top
    subset=np.concatenate((top,retained));order=np.sort(r2[subset]);local=r2.copy()
    strength=weakest[subset];slots=np.argsort(strength,kind='stable')
    for x in set(strength):
        mask=strength==x; local[subset[mask]]=order[mask[slots]].mean()
    return np.flatnonzero(local==max(local))

def summarize(table):
    weights=np.random.default_rng(281128).multinomial(len(table),np.full(len(table),1/len(table)),size=20000)
    return {'mean':float(np.mean(table)),'ci':np.quantile(weights@table/len(table),[.025,.975]).tolist()}

def collect(folder):
    allrows=read(folder/'inputs/rows.json');vectors=load(folder/'inputs/vectors.pt'); proof=read(folder/'scoring_complete.json')
    groups=defaultdict(list)
    for row in allrows:
        n=row['image'];scored=folder/'scores'/(n+'.json')
        assert sha(scored)==proof['files'][n]
        q=read(scored)
        if not q['eligible']:continue
        p=folder/'readouts'/(n+'.pt');assert sha(p)==q['projections']==proof['readouts'][n]
        raw=load(p);assert raw['rgb']==row['rgb_sha256']==q['rgb'] and raw['source']==q['source']
        assert sha(folder/'raw'/q['seed']/(n+'.pt'))==q['source']
        v=raw['visual'].double().numpy();t=vectors[n].double().numpy()
        assert v.shape==(8,768) and t.shape[0] in (8,12)
        sim=v@t.T
        dif=sim[:,::2]-sim[:,1::2]
        for a,item in zip(dif,q['items'],strict=True):
            np.testing.assert_allclose(a,item['hard_margins']+item['DE']+(item['FG'] or []),rtol=0,atol=1e-13)
        groups[row['pair']].append(dict(row=row,v=v,words=t,scores=sim,items=q['items'],base=q['r2']))
    return sorted((k,sorted(group,key=lambda a:a['row']['side'])) for k,group in groups.items() if len(group)==2)

def choose(im,partner):
    base=base_reward(im['items']); np.testing.assert_array_equal(base,im['base'])
    top=np.flatnonzero(base==max(base)); pos=im['words'][[0,2]]
    score=im['scores']; hard=score[:,:4:2]-score[:,1:4:2]
    variants={'strict':hard}
    for key,vv in [('bidirectional',partner['v']),('self',im['v'])]:
        variants[key]=np.column_stack((hard,score[:,:4:2]-vv.mean(axis=0)@pos.T))
    rr=np.array([v['other_caption_rr'] for v in im['items']])
    assert rr.shape[1]>=1 and rr.shape[1]==len(im['row']['alt_ids'])
    exact=[[Fraction.from_float(float(n)) for n in candidates] for candidates in rr]
    avg=[sum(exact[i][j] for i in top)/len(top) for j in range(rr.shape[1])]
    weights={k:np.zeros(8) for k in ARMS}; hidden=defaultdict(float);changed=defaultdict(float)
    for j in range(rr.shape[1]):
        visible=[c for c in range(rr.shape[1]) if c!=j]
        guard=[all(u[c]>=avg[c] for c in visible) for u in exact] if visible else [False]*8
        for key in ARMS:
            winners=top if key=='r2' else protected_winners(base,variants[key],guard)
            mass=1/(len(winners)*rr.shape[1]);weights[key][winners]+=mass
            hidden[key]+=float(rr[winners,j].sum()*mass)
            changed[key]+=(set(winners)!=set(top))/rr.shape[1]
    return weights,hidden,changed

def main():
    allstats={};used={}
    for source,p in SOURCES.items():
        records={a:defaultdict(list) for a in ARMS};groups=collect(p);used[source]=set()
        for pair,g in groups:
            left,right=g
            assert [r['row']['side'] for r in g]==[0,1]
            assert g[0]['row']['D']==g[1]['row']['D'][::-1] and g[0]['row']['E']==g[1]['row']['E'][::-1]
            for s in ('D','E'):
                assert len(g[0]['row'][s])==2
            side=[choose(left,right),choose(right,left)];seen={a:defaultdict(list) for a in ARMS}
            for j,a in enumerate(g):
                used[source].add(a['row']['rgb_sha256'])
                b=g[1-j]; weights,rr,changed=side[j]
                for arm in ARMS:
                    seen[arm]['MRR'].append(rr[arm]);seen[arm]['changed'].append(changed[arm])
                    prob=np.outer(weights[arm],side[1-j][0][arm]); assert abs(prob.sum()-1)<1e-12
                    for label,offset in [('DE',4),('FG',8)]:
                        if offset>=a['words'].shape[0]:continue
                        cols=[offset,offset+2];ours=a['scores'][:,cols];false=a['scores'][:,np.array(cols)+1]
                        i2t=np.mean((ours>false),axis=1)
                        theirs=b['v']@a['words'][cols].T
                        t2i=(ours[:,None,:]>theirs[None,:,:]).astype(float)+0.5*(ours[:,None,:]==theirs[None,:,:])
                        seen[arm][label+'_i2t'].append(float(weights[arm]@i2t))
                        seen[arm][label+'_t2i'].append(float((prob*t2i.mean(axis=2)).sum()))
            for arm,d in seen.items():
                for k,vals in d.items():
                    assert len(vals)==2
                    records[arm][k].append(np.mean(vals))
        stats={}
        for arm,other in [('r2',None),('strict',None),('bidirectional',None),('self',None),('bidirectional','strict'),('bidirectional','self'),('bidirectional','r2')]:
            name=arm if other is None else arm+'-'+other;stats[name]={}
            for k,values in records[arm].items():
                x=np.asarray(values)
                if other:x=x-np.asarray(records[other][k])
                stats[name][k]=dict(**summarize(x),pairs=len(x))
        allstats[source]=dict(pairs=len(groups),cases=len(used[source]),metrics=stats)
    assert [len(used[source]) for source in SOURCES]==[98,114] and not used['C98']&used['D114']
    c=allstats['C98']['metrics']['bidirectional-strict'];d=allstats['D114']['metrics']['bidirectional-strict']
    rules=dict(D114_LCB=d['DE_t2i']['ci'][0]>0,C98_gain=c['DE_t2i']['mean']>0,C98_i2t=c['DE_i2t']['mean']>=0,D114_i2t=d['DE_i2t']['mean']>=0,
               C98_MRR=c['MRR']['mean']>=0,D114_MRR=d['MRR']['mean']>=0,D114_over_self=allstats['D114']['metrics']['bidirectional-self']['DE_t2i']['mean']>0)
    result=dict(panels=allstats,conditions=rules,advance=all(rules.values()))
    (ROOT/'audit/双向奖励候选独立复算.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(rules)
if __name__=='__main__':
    torch.set_num_threads(1)
    main()
