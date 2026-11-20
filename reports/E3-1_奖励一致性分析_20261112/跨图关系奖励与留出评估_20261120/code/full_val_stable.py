"""Compute each unique-caption dot product once, then expand the original 1050 caption ownerships."""
import csv,math
from functools import lru_cache
import numpy as np
import torch
from shared import ROOT,read,sha,dump
from analyze_split import csvout
@lru_cache(None)
def success(total,correct,slots):
    return 1-math.comb(total-correct,slots)/math.comb(total,slots)
def recalls(a,labels):
    out={}
    for k in (1,5,10):
        cut=np.partition(a,-k,axis=1)[:,-k,None]
        greater=a>cut;tie=a==cut
        guaranteed=(greater&labels).any(axis=1)
        counts=tie.sum(1);pos=(tie&labels).sum(1);slots=k-greater.sum(1)
        values=[1. if sure else success(int(n),int(p),int(s)) for sure,n,p,s in zip(guaranteed,counts,pos,slots)]
        out[k]=float(np.mean(values))
    return out
def main():
    out=ROOT/'val';buf=torch.load(out/'features.pt',weights_only=True,map_location='cpu')
    arch=torch.load(out/'archive_features.pt',weights_only=True,map_location='cpu')
    oldout=read(out/'full_caption.json');order=arch['raw_to_column'];assert len(order)==1050
    native=buf['native'].double().numpy()
    def unique_scores(visual):
        # Fixed accumulation order independent of BLAS kernel and raw column offset.
        im=visual.double().numpy()
        sim=np.zeros((len(im),len(native)),dtype=np.float64)
        for i in range(im.shape[1]):sim+=im[:,i,None]*native[None,:,i]
        return sim[:,order]
    archive=unique_scores(arch['visual']);candidates=unique_scores(buf['visual'])
    owner=np.arange(210)[:,None]==np.repeat(np.arange(210),5)[None,:]
    def evaluate(scores):
        scores=np.ascontiguousarray(scores)
        d={'I2T_R'+str(k):v for k,v in recalls(scores,owner).items()}
        d.update({'T2I_R'+str(k):v for k,v in recalls(scores.T,owner.T).items()})
        d['mR']=float(np.mean(list(d.values())));return d
    ix=read(out/'index.json');rows=read(out/'rows.json')
    meta={(r['image'],r['mode'],r['k']):i for i,r in enumerate(rows)}
    with (out/'images.csv').open(encoding='utf-8-sig') as fp:groups=[r for r in csv.DictReader(fp) if r['mode']=='scene']
    draws=np.random.default_rng(20261120).random((64,len(groups)))
    result=[];stats={}
    for method in ('random','old','AB','greedy'):
        options=[[int(s) for s in r[method+'_k'].split(',')] for r in groups]
        values=[]
        for draw,rand in enumerate(draws[:1] if all(len(q)==1 for q in options) else draws):
            a=archive.copy()
            for j,r in enumerate(groups):
                k=options[j][int(rand[j]*len(options[j]))]
                a[ix['names'].index(r['image'])]=candidates[meta[r['image'],'scene',k]]
            metrics=evaluate(a);values.append(metrics);result.append(dict(rule=method,draw=draw,**metrics))
        stats[method]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    baseline=evaluate(archive)
    csvout(out/'full_caption_stable_draws.csv',result)
    dump(out/'full_caption_stable.json',dict(images=210,captions=1050,replaced=len(groups),unique_caption_embeddings=len(native),
         archive=baseline,selectors=stats,old_archive_mR=oldout['archive']['mR'],archive_features_hash=sha(out/'archive_features.pt'),
         input_features_hash=sha(out/'features.pt'),draws=64))
if __name__=='__main__':main()
