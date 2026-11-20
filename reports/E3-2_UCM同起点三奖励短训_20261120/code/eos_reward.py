"""Post-hoc test on a shared pre-update candidate pool for preference for immediate EOS."""
import hashlib,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from shared import ROOT,read,dump,sha
from analysis import csvwrite
def main(prior):
    cases=read(ROOT/'inputs/train.json')
    v=torch.load(prior/'scoring/vectors.pt',weights_only=True,map_location='cpu');rr=read(prior/'scoring/rows.json')
    ii=read(prior/'native/index.json');f=torch.load(prior/'native/features.pt',weights_only=True,map_location='cpu')
    assert sha(prior/'native/features.pt')==ii['sha256'] and sha(prior/'scoring/vectors.pt')==read(prior/'scoring/binding.json')['bank']
    parity=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in f['tokens'].tolist()])
    allrows=[];raw=[]
    def metrics(x,e):
        a=x[e];b=x[~e];delta=a[:,None]-b
        return dict(win=float(np.mean((delta>0)+.5*(delta==0))),eos_selected=float(e[x==x.max()].mean()),eos_random=float(e.mean()))
    for i,r in enumerate(cases):
        n=r['image'];g=[j for j,z in enumerate(rr) if (z['image'],z['mode'])==(n,'scene') and z['k']>=0]
        assert [rr[j]['k'] for j in g]==list(range(8))
        e=[]
        for k,j in enumerate(g):
            z=read(prior/'samples/scene'/Path(n).stem/f'sample{k}.json')
            assert z['query']==r['scene_query'] and rr[j]['native_tensor']==z['vector_sha256']
            e.append(z['rounds'][0]['ids'] in [[151645],[151643]])
        e=np.array(e)
        mp=ii['maps'][n];mask=parity.copy();mask[mp['positive']]=False
        s=v['visual'][g].double().numpy()@f['features'].double().numpy().T
        a=rankdata(s[:,mp['first']]-s[:,mask].max(1))
        b=rankdata([rr[j]['A'] for j in g])
        methods={'old':np.array([rr[j]['reward'] for j in g]),'caption':a,'hard':(a+b)/2}
        for k,j in enumerate(g):raw.append(dict(image=n,pair=r['pair'],k=k,eos=bool(e[k]),**{arm:float(x[k]) for arm,x in methods.items()}))
        for arm in ('old','caption','hard'):
            for pool in ('shared_base','during_training'):
                if pool=='shared_base':x=methods[arm];labels=e
                else:
                    z=read(ROOT/('policy_'+arm)/'steps.json')[i];assert z['image']==n
                    x=np.array(z['rewards']);labels=np.array([tr['rounds'][0]['tokens'] in [[151645],[151643]] for tr in z['traces']])
                if labels.any() and not labels.all():allrows.append(dict(pool=pool,image=n,pair=r['pair'],arm=arm,**metrics(x,labels)))
    summaries={}
    for pool in ('shared_base','during_training'):
        for arm in ('old','caption','hard'):
            group=[r for r in allrows if r['pool']==pool and r['arm']==arm]
            pairs=sorted({r['pair'] for r in group});counts=np.array([sum(z['pair']==p for z in group) for p in pairs]);rng=np.random.default_rng(20261120)
            selected=rng.integers(len(pairs),size=(20000,len(pairs)))
            values={}
            for key in ('win','eos_selected','eos_random'):
                totals=np.array([sum(z[key] for z in group if z['pair']==p) for p in pairs])
                lo,hi=np.quantile(totals[selected].sum(1)/counts[selected].sum(1),[.025,.975])
                values[key]=dict(mean=float(totals.sum()/counts.sum()),lo=float(lo),hi=float(hi))
            summaries[pool+'/'+arm]=dict(mixed_images=len(group),pairs=len(pairs),metrics=values)
    csvwrite(ROOT/'results/eos_comparisons.csv',allrows);csvwrite(ROOT/'results/eos_shared_candidates.csv',raw);dump(ROOT/'results/eos_reward.json',summaries)
if __name__=='__main__':main(Path(sys.argv[1]))
