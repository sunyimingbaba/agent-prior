"""Reproduce all primary selected sets from saved features and independent tie handling."""
import csv,hashlib,json,sys
import numpy as np
import torch
from analyze import ROOT,RECORDS,OUT,files,tensor,read,sha,dump
def sorted_groups(values):
    pool={}
    for j,v in enumerate(values):pool.setdefault(v,[]).append(j)
    return [s for _,s in sorted(pool.items())]
def assign(values,scale):
    out=np.zeros(len(values));start=0
    for group in sorted_groups(values):
        out[group]=np.mean(scale[start:start+len(group)])
        start+=len(group)
    return out
def main(mode='main'):
    out=OUT if mode=='main' else OUT/mode
    p=read(RECORDS/'provenance.json')
    assert p['tools']==sha(RECORDS/'tools.json')
    bank=torch.load(RECORDS/'text.pt',weights_only=True,map_location='cpu')
    native=bank['native'].numpy().astype(float);hard=bank['hard'].numpy().astype(float)
    mask0=np.array([int.from_bytes(hashlib.sha256(','.join(str(int(t)) for t in v).encode()).digest(),'big')%2==0 for v in bank['native_tokens'].numpy()])
    with (out/'choices.csv').open(encoding='utf-8-sig') as f:
        cs={(r['arm'],r['image'],r['method']):r for r in csv.DictReader(f)}
    continuous=out/'continuous_choices.csv'
    cc={};kappa=None
    if continuous.exists():
        with continuous.open(encoding='utf-8-sig') as f:cc={(r['arm'],r['image']):r for r in csv.DictReader(f)}
        lock=next(p for p in (ROOT/'inputs/continuous_lock.json',ROOT.parent/'inputs/continuous_lock.json') if p.is_file())
        kappa=read(lock)['kappa']
    err=0.;checks=0;additional=0
    for arm in ('a','b'):
        banks=[arm] if mode=='main' else ['extra_'+arm] if mode=='extra' else [arm,'extra_'+arm]
        data={}
        for prefix in banks:data.update(files(prefix))
        for row in read(ROOT/'inputs/panel.json'):
            nm=row['image'];objs=[tensor(data[f'{prefix}_readouts/{nm}.pt']) for prefix in banks]
            x=torch.cat([obj['visual'] for obj in objs]).double().numpy()
            ix=bank['maps'][nm];mask=mask0.copy();mask[ix['native']]=False
            sc=x@native.T
            m=sc[:,ix['native'][0]]-sc[:,mask].max(1)
            marg={k:x@(hard[ix[k][0]]-hard[ix[k][1]]) for k in ix if k in 'ABCD'}
            A=marg['A']
            hard_rank=assign(A,np.arange(1,len(x)+1));nr=assign(m,np.arange(1,len(x)+1))
            proxy=sorted(1/(1+float(rank)) for obj in objs for rank in obj['proxy_rank'])
            ev=(~mask0);ev[ix['native']]=False
            best=sc[:,sorted(set(ix['native'])-{ix['native'][0]})].max(1)
            rr=[]
            for z,t in zip(sc,best):
                hi=int((z[ev]>t).sum());tie=int((z[ev]==t).sum())
                rr.append(sum(1/(1+hi+j) for j in range(tie+1))/(tie+1))
            methods=[('R2',m),('R3',3*nr+hard_rank),('R3_50',nr+hard_rank)]
            if cc:methods.append(('continuous',m+kappa*A))
            for method,v in methods:
                z=assign(v,proxy);pick=np.flatnonzero(z==max(z))
                old=cc[arm,nm] if method=='continuous' else cs[arm,nm,method]
                assert list(pick)==list(map(int,old['slots'].split(',')))
                for key,vec in [(k,v>0) for k,v in marg.items()]+[('MRR',np.array(rr))]:
                    if key in old:err=max(err,abs(float(old[key])-float(vec[pick].mean())))
                if method=='continuous':additional+=1
                else:checks+=1
    assert err<1e-9
    dump(out/'verification.json',dict(checked_R2_R3_25_R3_50_sets=checks,checked_continuous_sets=additional,max_recomputed_metric_difference=err,
                                      verified_archives=2*len(banks)))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'main')
