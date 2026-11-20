"""Measure proxy-selection transfer to the independent native-caption retrieval gallery."""
from pathlib import Path
import re,collections
import numpy as np
import torch
from shared import ROOT,read,dump,sha
from analyze import save_csv,spearman,stats
out=ROOT/'native';ix=read(out/'index.json')
assert ix['sha256']==sha(out/'features.pt')
f=torch.load(out/'features.pt',weights_only=True,map_location='cpu')
fts=torch.load(ROOT/'scoring/vectors.pt',weights_only=True,map_location='cpu')
rows=read(ROOT/'scoring/rows.json')
sims=fts['visual'].double().numpy()@f['features'].double().numpy().T
assert sims.shape==(len(rows),ix['token_groups']) and ix['raw']==8400
result=[];table=[]
for i,z in enumerate(rows):
    qs=sims[i];m=ix['maps'][z['image']]
    neg=np.ones(len(qs),dtype=bool);neg[m['positive']]=False
    bestneg=float(qs[neg].max())
    for name in ('all','drop_first'):
        positive=[q for q in m['positive'] if name=='all' or q!=m['first']]
        if not positive:continue
        best=float(qs[positive].max())
        rank=int(np.sum(qs[neg]>best))
        table.append(dict(image=z['image'],mode=z['mode'],pair=z['pair'],k=z['k'],gallery=name,
                          reward=z['reward'],A=z['A'],B=z['B'],rank0=rank,rr=1/(1+rank),margin=best-bestneg,
                          best_positive=best,best_negative=bestneg))
save_csv('训练原描述逐候选.csv',table)
for (mode,tag,n),gr in __import__('itertools').groupby(sorted([r for r in table if r['k']>=0],
            key=lambda x:(x['mode'],x['gallery'],x['image'],x['k'])),key=lambda x:(x['mode'],x['gallery'],x['image'])):
    g=list(gr);assert len(g)==8
    a,b,proxy,rr,mar=[np.array([r[key] for r in g]) for key in ('A','B','reward','rr','margin')]
    x=dict(mode=mode,gallery=tag,image=n,pair=g[0]['pair'],proxy_rho=spearman(proxy,mar),A_rho=spearman(a,mar),B_rho=spearman(b,mar))
    for metric,values in [('rr',rr),('margin',mar)]:
        mean=float(values.mean());x['random_'+metric]=mean
        for method,sel in [('proxy',proxy),('A',a),('B',b)]:
            val=float(values[sel==max(sel)].mean())
            x[method+'_'+metric]=val;x[method+'_gain_'+metric]=val-mean
    result.append(x)
save_csv('训练原描述逐图.csv',result)
summ={}
for mode in ('scene','original'):
    summ[mode]={}
    for tag in ('all','drop_first'):
        g=[r for r in result if r['gallery']==tag and r['mode']==mode]
        keys=[k for k in g[0] if k not in ('mode','gallery','image','pair')]
        summ[mode][tag]=dict(n=len(g),stats={k:stats(g,k) for k in keys})
dump(ROOT/'训练原描述汇总.json',summ)
print(ix['raw'],ix['token_groups'],len(table))
