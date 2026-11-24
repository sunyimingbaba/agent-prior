"""Rebuild paired retrieval and attribute endpoints from archived FP32 vectors."""
import json
from collections import defaultdict
from itertools import chain
import numpy as np
import torch
from shared import ROOT,read,dump,sha
from metrics import fixed,recalls,csvwrite

STATES=['R0','a_R1','b_R1','a_R2','b_R2','a_first','b_first']
ARMS={'R0':['R0','R0'],'R1':['a_R1','b_R1'],'R2':['a_R2','b_R2'],'first':['a_first','b_first']}
CONTRAST=[('R2','R1'),('R2','R0'),('R2','first'),('R1','R0')]
OUT=ROOT/'results'
def load(name):return torch.load(OUT/name,weights_only=True,map_location='cpu')
def hit_sums(img,txt,owners,n,valid):
    counts=np.bincount(owners,minlength=n)[valid]
    sums=np.bincount(owners,weights=txt,minlength=n)[valid]
    return np.column_stack([img,np.ones(len(valid)),sums,counts])
def value(x):
    return 50*(x[...,0]/x[...,1]+x[...,2]/x[...,3])
def bootstrap(matrix,groups=None):
    if groups is not None:
        matrix=np.stack([matrix[groups==g].sum(0) for g in np.unique(groups)])
    n=len(matrix);rng=np.random.default_rng(20261120)
    dist=[]
    for _ in range(40):
        ids=rng.integers(n,size=(500,n));dist.extend(value(matrix[ids].sum(1)))
    lo,hi=np.quantile(dist,[.025,.975])
    return dict(mean=float(value(matrix.sum(0))),lo=float(lo),hi=float(hi),clusters=n)
def difference(a,b):
    assert np.array_equal(a[:,[1,3]],b[:,[1,3]])
    z=a.copy();z[:,[0,2]]-=b[:,[0,2]]
    return z
def contrasts(states,groups=None):
    out={}
    for a,b in CONTRAST:
        ds=[difference(states[x],states[y]) for x,y in zip(ARMS[a],ARMS[b])]
        d=np.mean(ds,axis=0)
        result=bootstrap(d)
        result['per_seed']=[float(value(z.sum(0))) for z in ds]
        if groups is not None:result['scene_bootstrap']=bootstrap(d,groups)
        out[a+'-'+b]=result
    return out
def retrieval(dn):
    cases=read(ROOT/'inputs'/f'{dn}.json');gallery=load(dn+'_text.pt')
    all_order=gallery['caption_order'].numpy().reshape(-1,5);n=len(cases)
    assert all_order.shape==(n,5)
    h=sha(ROOT/'inputs'/f'{dn}.json')
    screen=set(read(ROOT/'inputs/overlap_screen.json')['exclude'][dn])
    keep=np.array([i for i,c in enumerate(cases) if c['image'] not in screen])
    subsets={'all':np.arange(n),'overlap_filtered':keep}
    scores={};features={}
    for state in STATES:
        row=load(f'{dn}_{state}.pt')
        assert row['manifest']==h and row['rows']==[c['row'] for c in cases]
        features[state]=row['features']
        scores[state]=fixed(features[state],gallery['native'])
    summary={};details=[]
    for subset,ii in subsets.items():
        if len(ii)<10:continue
        order=all_order[ii];ns=len(ii);ids=np.arange(ns)
        scene=np.array([(int(cases[i]['image'].split('.')[0])-1)//100 for i in ii]) if dn=='ucm' else None
        for mode in ('native','alternatives'):
            if subset=='overlap_filtered' and mode!='native':continue
            captions=np.arange(5*ns) if mode=='native' else np.flatnonzero((order!=order[:,:1]).ravel())
            owners=captions//5;valid=np.unique(owners)
            labels=ids[:,None]==owners
            states={};values={}
            for state in STATES:
                sim=scores[state][ii][:,order.ravel()[captions]]
                i2t=sim.copy()
                if mode=='alternatives':
                    i2t[order[:,0,None]==order.ravel()[captions][None,:]]=-np.inf
                r=recalls(i2t[valid],labels[valid]);t=recalls(sim.T,labels.T)
                states[state]=hit_sums(np.mean(list(r.values()),axis=0),np.mean(list(t.values()),axis=0),owners,ns,valid)
                values[state]={f'I2T_R{k}':100*float(v.mean()) for k,v in r.items()}
                values[state].update({f'T2I_R{k}':100*float(v.mean()) for k,v in t.items()})
                values[state]['mR']=float(value(states[state].sum(0)))
                assert np.isclose(np.mean(list(values[state].values())[:6]),values[state]['mR'])
                for j,(ih,ic,th,tc) in zip(valid,states[state]):
                    details.append(dict(domain=dn,subset=subset,mode=mode,arm=state,image=cases[ii[j]]['image'],
                                        image_mean_hit=ih,text_sum_hit=th,text_count=int(tc)))
            summary[subset+'_'+mode]=dict(image_gallery=ns,source_images=len(valid),captions=len(owners),
              levels=values,differences=contrasts(states,scene[valid] if scene is not None else None))
    csvwrite(OUT/f'{dn}_retrieval.csv',details)
    return summary,features

def attributes(features):
    cases=read(ROOT/'inputs/ucm.json');attrs=read(ROOT/'inputs/attributes.json');bank=load('attribute_text.pt')
    assert bank['manifest']==sha(ROOT/'inputs/attributes.json')
    w={x:i for i,x in enumerate(bank['texts'])};im={c['image']:i for i,c in enumerate(cases)}
    selected={'all':attrs,'old_extension':[r for r in attrs if r['vocabulary']=='old_extension']}
    q={r['image']:r['orig_query'] for r in cases}
    pairs=defaultdict(list)
    for r in attrs:pairs[r['pair']].append(r)
    selected['identical_input_text']=[r for pair in pairs.values() if len({q[r['image']] for r in pair})==1 for r in pair]
    all_rows=[];out={}
    for group,rows in selected.items():
        pairs=list(dict.fromkeys(r['pair'] for r in rows))
        states={};levels={};counts={}
        for state in STATES:
            sc=fixed(features[state],bank['features'])
            grouped=defaultdict(list)
            for r in rows:
                j=im[r['image']]
                margin=sc[j,w[r['positive']]]-sc[j,w[r['negative']]]
                grouped[r['pair']].append(int(margin>0))
                if group=='all':all_rows.append(dict(state=state,**r,margin=margin,correct=int(margin>0)))
            bypair=np.array([grouped[p] for p in pairs]);assert bypair.shape==(len(pairs),2)
            counts[state]=bypair
            correct=bypair.sum(1)
            levels[state]=dict(correct=int(correct.sum()),both=int((correct==2).sum()),ties=int(sum(r['margin']==0 for r in all_rows if r['state']==state and r['pair'] in pairs)))
            states[state]=np.column_stack([correct,np.full(len(pairs),2),correct,np.full(len(pairs),2)])
        scenes=np.array([next(r['scene'] for r in rows if r['pair']==pair) for pair in pairs])
        flip={}
        for b in ['R0','R1','first']:
            flip[b]=[dict(gained=int(((counts[x]==1)&(counts[y]==0)).sum()),lost=int(((counts[x]==0)&(counts[y]==1)).sum())) for x,y in zip(ARMS['R2'],ARMS[b])]
        out[group]=dict(items=len(rows),pairs=len(pairs),levels=levels,differences=contrasts(states,scenes),r2_flips=flip)
    csvwrite(OUT/'attribute_by_image.csv',all_rows)
    return out
if __name__=='__main__':
    info={};u=None
    for dn in ('ucm','rsitmd'):
        info[dn],features=retrieval(dn)
        if dn=='ucm':u=features
    info['attributes']=attributes(u)
    dump(OUT/'paired_results.json',info)
