"""Build gain features from the visible labels, with held targets stored separately."""
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
GUARD=ROOT.parent/'E3_R3首位配文保护与首句复验_20261126'
PREV=ROOT.parent/'E3_R3跨措辞与相似图对复验_20261126'
sys.path.insert(0,str(GUARD/'code'))
from guard import PRIOR, PRIOR_RUN, OLD, best, read, sha, rr, load_bank, previous_panel, previous_banks, real_stream
from 候选复验 import summarize

ROTATIONS=((0,1,2),(0,2,1),(1,2,0))
ARMS=('ordinary','old','old3','switch_ab','switch','rank_only')
LIBRARIES={'v122':PRIOR/'guard','v148':ROOT.parent/'E3_UCM多数配文新图复核_20261125'}


def stream(tag):
    if tag=='history98':
        yield from real_stream()
        return
    if tag in ('dev','history78'):
        held={r['rgb_sha256'] for r in read(PRIOR_RUN/'inputs/holdout.json')}
        rows=[r for r in read(OLD/'inputs/real_roster.json') if (r['rgb_sha256'] in held)==(tag=='history78')]
        yield from previous_panel(rows,previous_banks())
        return
    source=LIBRARIES[tag]
    rows=[r for r in read(ROOT/'inputs/real_roster.json') if r['cohort']==tag]
    bank=load_bank(source)
    for seed in 'ab':
        base=ROOT/'records/real'/seed
        manifest=read(base/'raw/complete.json')
        verify={k:h for m in sorted((base/'features').glob('complete*.json')) for k,h in read(m)['files'].items()}
        assert manifest['adapter']==read(source/'records/provenance.json')['adapters'][seed]
        for row in rows:
            n=row['image'];pf=base/'features'/(n+'.pt');pr=base/'raw'/(n+'.pt')
            assert sha(pf)==verify[n] and sha(pr)==manifest['files'][n]
            z=torch.load(pf,map_location='cpu',weights_only=True);r=torch.load(pr,map_location='cpu',weights_only=True)
            assert z['adapter']==r['adapter']==manifest['adapter'] and z['rgb']==r['rgb']==row['rgb_sha256']
            assert r['query']==row['orig_query']==row['captions'][0] and r['feedback']==row['feedback']
            assert r['rngs']==[row['rng_'+seed],row['rng_'+seed]+100] and z['source']==manifest['files'][n]
            assert len(r['paths'])==16 and z['visual'].shape==(17,768)
            yield row,seed,'first',bank,z['visual'].double().numpy()


def centered(v,ref):
    return v-v[ref].mean(0)


def target(correct,margin,held,ref):
    df=correct[:,None]-correct[ref]
    dr=held[:,None]-held[ref]
    net=((df>0)&(dr>0)).mean(1)-((df<0)&(dr<0)).mean(1)
    return np.column_stack((df.mean(1),centered(margin,ref),dr.mean(1),net))


def gains_features(h,reference,similar,rrs,gaps,first,near,spread,base):
    # Per-phrase order is intentionally symmetric in the new model.
    cross=centered(h,reference)
    positive=h>0
    pair=np.column_stack((h.min(1),h.max(1),h.mean(1),positive.mean(1),
                          cross.min(1),cross.max(1),cross.mean(1),(cross>0).mean(1),
                          (centered(positive.astype(float),reference)).mean(1)))
    def stats(values):
        delta=centered(values,reference)
        return np.column_stack((values.mean(1),values.min(1),delta.min(1),delta.max(1),
                                delta.mean(1),(delta>1e-12).mean(1),(delta< -1e-12).mean(1)))
    common=np.column_stack((stats(rrs),stats(gaps),stats(similar),stats(near),first,
                            np.full(8,spread),base))
    combined=np.column_stack((pair,common))
    relative=centered(combined,reference)
    # Give the linear model access to baseline difficulty near either decision boundary.
    baseline_margin=abs(h[reference].mean())
    certainty=np.exp(-baseline_margin/.025)
    room=1-rrs[reference].mean()
    extras=np.column_stack((cross.mean(1)*certainty,cross.min(1)*certainty,
                            centered(gaps.mean(1),reference)*room,centered(near.mean(1),reference)*room))
    return np.column_stack((combined,relative,extras))


def packets(source):
    for row,seed,view,bank,v in source:
        ix=bank['maps'][row['image']]
        col=ix['native'];alts=sorted(set(col)-{col[0]})
        if len(alts)<2:continue
        hard_ids=np.array([ix[k] for k in 'ABCDE'])
        assert set(hard_ids[:3].ravel()).isdisjoint(hard_ids[3:].ravel())
        hard=v[:16]@bank['hard'][hard_ids.ravel()].T
        h=hard[:,::2]-hard[:,1::2]
        vis=v[:16]@bank['native'].T
        odd=~bank['even'].copy();even=bank['even'].copy();odd[col]=False;even[col]=False
        pos=vis[:,alts]
        neg=vis[:,odd]
        rr_fit=np.array([[rr(r[k],r[odd]) for k in alts] for r in vis],dtype=object)
        rr_test=np.array([[float(rr(r[k],r[even])) for k in alts] for r in vis])
        eight=np.partition(neg,-8,axis=1)[:,-8:]; worst=eight.max(1)
        text=bank['native'][alts]
        neighbor=text@bank['native'][odd].T
        close=np.argsort(-neighbor,axis=1,kind='stable')[:,:8]
        near=pos-neg[:,close].mean(2)
        fst=vis[:,col[0]]
        fst_rank=np.array([float(rr(x,g)) for x,g in zip(fst,neg)])
        for held in range(len(alts)):
            cols=[j for j in range(len(alts)) if j!=held]
            ranking=rr_fit[:,cols].mean(1)
            rank_values=rr_fit[:,cols].astype(float)
            visible=pos[:,cols]
            g=visible-worst[:,None]
            first=np.column_stack((fst,fst-worst,fst_rank))
            spread=float(np.mean(text[cols]@bank['native'][col[0]]))
            for block,start in enumerate((0,8)):
                sl=slice(start,start+8)
                ref=best(ranking[sl],range(8))
                ordinary=ranking[sl].astype(float)
                hard=h[sl]; held_rr=rr_test[sl,held]
                xs=[];ys=[];old=[]
                for a,b,c in ROTATIONS:
                    phrase=hard[:,[a,b]]
                    base=np.column_stack((ordinary,rank_values[sl].min(1),
                        visible[sl].mean(1),visible[sl].min(1),g[sl].mean(1),g[sl].min(1),
                        visible[sl].mean(1)-eight[sl].mean(1)))
                    oldx=np.column_stack((phrase[:,0],phrase[:,1],phrase.min(1),
                                    phrase[:,0]>0,phrase[:,1]>0,(phrase>0).all(1),base))
                    old.append(oldx-oldx.mean(0))
                    xs.append(gains_features(phrase,ref,visible[sl],rank_values[sl],g[sl],first[sl],near[sl][:,cols],spread,base))
                    ys.append(target((hard[:,c]>0).astype(float),hard[:,c],held_rr,ref))
                yield dict(image=row['image'],rgb=row['rgb_sha256'],pair=row['pair'],view=view,seed=seed,held=held,
                           block=block,w=1/len(alts),ref=ref,ordinary=ordinary,exact=ranking[sl],
                           x=np.stack(xs),y=np.stack(ys),old=np.stack(old),fine=(hard[:,3:]>0).mean(1),
                           c=(hard[:,2]>0).astype(float),mrr=held_rr)


def outcomes(p,tops,dev=False):
    fine=p['c'] if dev else p['fine'];rank=p['mrr'];ref=p['ref']
    good=(fine>fine[ref].mean()+1e-12)&(rank>rank[ref].mean()+1e-12)
    result={}
    for arm,ws in tops.items():
        changed=ws!=ref
        dif=fine[ws,None]-fine[ref];dr=rank[ws,None]-rank[ref]
        win=float(((dif>0)&(dr>0)).mean()) if changed else 0.
        loss=float(((dif<0)&(dr<0)).mean()) if changed else 0.
        result[arm]=dict(top=ws,detail=float(fine[ws].mean()),mrr=float(rank[ws].mean()),net=win-loss,
                         changed=changed,win=win,loss=loss)
    return {k:p[k] for k in ('image','rgb','pair','view','seed','held','block')} | dict(
        support=0,oracle=bool(good.any()),outcomes=result)
