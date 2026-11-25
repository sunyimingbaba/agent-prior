"""Reproduce both the ordinary and alternate-caption retrieval protocols on a fixed gallery."""
import sys
import numpy as np
from rosters import ROOT,read,sha,dump
from single_scores import FORM,RFPO,fixed,recalls,load

def counts(visual,gal,mode):
    n=len(visual);order=gal['caption_order'].numpy().reshape(n,5)
    scores=fixed(visual,gal['native'])
    which=np.arange(n*5) if mode=='native' else np.flatnonzero((order!=order[:,:1]).ravel())
    owners=which//5;valid=np.unique(owners);target=np.arange(n)[:,None]==owners[None,:]
    mat=scores[:,order.ravel()[which]];forward=mat.copy()
    if mode=='alternatives':forward[order[:,0,None]==order.ravel()[which][None,:]]=-np.inf
    f=recalls(forward[valid],target[valid]);t=recalls(mat.T,target.T)
    nf=np.stack(list(f.values()),axis=1);nt=np.stack([np.bincount(owners,weights=z,minlength=n)[valid] for z in t.values()],axis=1)
    denom=np.bincount(owners,minlength=n)[valid]
    return valid,np.column_stack([nf,nt]),denom

def scalar_values(totals,images,captions):
    vals=totals/np.stack([images]*3+[captions]*3,axis=-1)
    return np.concatenate((vals,vals[...,:3].mean(-1,keepdims=True),vals[...,3:].mean(-1,keepdims=True),vals.mean(-1,keepdims=True)),axis=-1)

def summary(arr,ds):
    n=len(ds);rng=np.random.default_rng(20261125);draws=[]
    for i in range(40):
        ids=rng.integers(n,size=(500,n))
        draws.append(scalar_values(arr[ids].sum(1),np.full(500,n),ds[ids].sum(1)))
    sample=np.concatenate(draws);value=scalar_values(arr.sum(0),np.asarray(n),np.asarray(ds.sum()))
    ci=np.quantile(sample,[.025,.975],axis=0)
    keys=('I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','I2T','T2I','mR')
    return {k:dict(mean=float(value[i]),interval=ci[:,i].tolist()) for i,k in enumerate(keys)}

def main(dn):
    gal=load(FORM/'results'/(dn+'_text.pt'));cases=read(FORM/'inputs'/(dn+'.json'))
    expected=read(FORM/'results/paired_results.json')[dn];result={};records=[]
    for mode in ('native','alternatives'):
        old=[];new=[];replay={};coverage={}
        for seed in 'ab':
            hist=load(FORM/'results'/(dn+'_'+seed+'_R2.pt'))
            if dn=='ucm':
                p=ROOT/'records/route'/seed/'official_features.pt';assert sha(p)==read(p.with_name('official_complete.json'))['features']
                v=load(p);ft=v['visual'];anchor=v['baseline'][cases[0]['image']]
            else:
                p=ROOT/'records/cross'/seed/'features.pt';proof=read(p.with_name('read.json'))
                assert sha(p)==proof['features'] and sha(p.with_name('trajectories.tar.gz'))==proof['archive'];v=load(p);ft=v['features'];anchor=v['baseline']
            assert [z['image'] for z in v['meta']]==[c['image'] for c in cases] and v['adapter']==hist['adapter'] and v['route']==sha(ROOT/'inputs/route.json')
            assert hist['manifest']==sha(FORM/'inputs'/(dn+'.json'))
            replay[seed]=float((anchor-hist['features'][0]).abs().max());assert replay[seed]<1e-4
            ids,b,ds=counts(hist['features'],gal,mode);ids1,a,ds1=counts(ft,gal,mode)
            assert np.array_equal(ids,ids1) and np.array_equal(ds,ds1)
            coverage[seed]=sum(bool(z['attribute']) for z in v['meta']);old.append(b);new.append(a)
            baseline=scalar_values(b.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))
            ref=expected['all_'+mode]['levels'][seed+'_R2']
            for i,k in enumerate(['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10']):
                assert abs(100*baseline[i]-ref[k])<1e-10
            assert abs(100*baseline[-1]-ref['mR'])<1e-10
            for j,idx in enumerate(ids):
                records.append(dict(mode=mode,seed=seed,image=cases[idx]['image'],text_count=int(ds[j]),old=b[j].tolist(),new=a[j].tolist()))
        change=np.subtract(new,old)
        result[mode]=dict(images=len(cases),eligible=len(ids),captions=int(ds.sum()),routed=coverage,replay=replay,
            baseline=summary(np.mean(old,0),ds),route=summary(np.mean(new,0),ds),change=summary(change.mean(0),ds),
            seeds={s:scalar_values(x.sum(0),np.asarray(len(ds)),np.asarray(ds.sum())).tolist() for s,x in zip('ab',change)})
    dump(ROOT/'results'/('single_'+dn+'.json'),result);dump(ROOT/'records'/('single_'+dn+'_images.json'),records)
if __name__=='__main__':main(sys.argv[1])
