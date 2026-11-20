"""Tie-aware within-image alignment between historical proxy, alternative reward and retrieval benefits."""
import sys,csv
import numpy as np
from scipy.stats import rankdata
import torch
from shared import ROOT,read,dump,sha
from analyze_split import csvout,estimate
def main(split):
    out=ROOT/split;index=read(out/'index.json');raw=read(out/'rows.json')
    f=torch.load(out/'features.pt',weights_only=True,map_location='cpu')
    with (out/'candidates.csv').open(encoding='utf-8-sig') as fp:rows=list(csv.DictReader(fp))
    assert len(rows)==len(raw)
    n=f['visual'].double().numpy()@f['native'].double().numpy().T
    for i,r in enumerate(rows):
        p=index['native_target'][r['image']]
        neg=np.ones(n.shape[1],bool);neg[p]=False
        r['original_caption_margin']=n[i,p].max()-n[i,neg].max()
    items=[];summ={}
    for mode in ('scene','original'):
        for name in sorted({r['image'] for r in raw}):
            g=[r for r in rows if r['image']==name and r['mode']==mode and int(r['k'])>=0];assert len(g)==8
            vals={key:np.array([float(r[key]) for r in g]) for key in ['A','B','C','D','old','original_caption_margin','RR']}
            rr={'AB':(rankdata(vals['A'])+rankdata(vals['B']))/2,'old':vals['old']}
            for ru,x in rr.items():
                for target in ('C','D','original_caption_margin','RR'):
                    y=vals[target];rho=None
                    if len(set(x))>1 and len(set(y))>1:rho=float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
                    items.append(dict(image=name,mode=mode,pair=g[0]['pair'],selector=ru,target=target,rho=rho))
    csvout(out/'within_image.csv',items)
    for mode in ('scene','original'):
        for selector in ('AB','old'):
            for target in ('C','D','original_caption_margin','RR'):
                cluster={}
                for z in items:
                    if (z['mode'],z['selector'],z['target'])==(mode,selector,target) and z['rho'] is not None:cluster.setdefault(z['pair'],[]).append(z['rho'])
                if cluster:
                    groups=list(cluster.values());count=np.array(list(map(len,groups)));sums=np.array(list(map(sum,groups)))
                    idx=np.random.default_rng(20261120).integers(len(groups),size=(20000,len(groups)))
                    ci=np.quantile(sums[idx].sum(1)/count[idx].sum(1),[.025,.975])
                    summ['/'.join((mode,selector,target))]=dict(mean=float(sums.sum()/count.sum()),lo=float(ci[0]),hi=float(ci[1]),images=int(count.sum()),pairs=len(groups))
    dump(out/'consistency.json',summ)
if __name__=='__main__':main(sys.argv[1])
