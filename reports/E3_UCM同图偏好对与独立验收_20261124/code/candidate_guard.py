"""Training-pool diagnostic. Native rank bands are defined by the retrieval cutoffs 1,5,10."""
import sys,hashlib,numpy as np,torch
from pathlib import Path
from scipy.stats import rankdata
from dev_metrics import ROOT,HIST,read,load,boot,dump,write_csv
sys.path.insert(0,str(HIST/'code'))
from verify_screen import unpack,OLD
from analyze import calibrated
def row_scores(native,hard,tokens,ix,vis,proxy,kappa):
    p=ix['native'];pos=np.unique(p);alt=np.setdiff1d(pos,p[0]);assert len(alt)
    allow=np.ones(len(tokens),dtype=bool);allow[pos]=False
    sim=vis.double().numpy()@native.T;hs=vis.double().numpy()@hard.T;train=sim[:,allow&tokens];held=sim[:,allow&~tokens]
    raw=sim[:,p[0]];m=raw-train.max(1);rank=1+(train>raw[:,None]).sum(1)
    band=np.searchsorted([1,5,10],rank,side='left');A=hs[:,ix['A'][0]]-hs[:,ix['A'][1]];r3=m+kappa*A
    order=rankdata(r3,method='dense')+(3-band)*(1+len(rank));sc={'R2':m,'R3':r3,'guard':order}
    best=sim[:,alt].max(1);rr=np.array([np.mean(1/np.arange(1+(s>v).sum(),2+(s>v).sum()+(s==v).sum())) for s,v in zip(held,best)])
    out={}
    for key,o in sc.items():
        calibrated_scores=calibrated(o,1/(1+np.array(proxy)));winner=np.flatnonzero(calibrated_scores==calibrated_scores.max())
        out[key]=dict(winner=winner.tolist(),rr=float(rr[winner].mean()),**{l:float((hs[winner,ix[l][0]]>hs[winner,ix[l][1]]).mean()) for l in ix if l in 'BCDE'})
    return out
def summarize(rows):
    res={}
    for left,right in [('R3','R2'),('guard','R2'),('guard','R3')]:
        ordered={s:[r for r in rows if r['seed']==s] for s in 'ab'};metrics={}
        for k in ('detail','rr'):
            ds=[]
            for s,v in ordered.items():
                def score(i,method):return np.mean([vv for key,vv in i['scores'][method].items() if key in ('B','C','D','E')]) if k=='detail' else i['scores'][method]['rr']
                ds.append([score(i,left)-score(i,right) for i in v])
            d=np.array(ds);metrics[k]=dict(**boot(d.mean(0),[r['pair'] for r in ordered['a']]),seeds=d.mean(1).tolist())
        res[left+'-'+right]=metrics
    return res
def main():
    torch.set_num_threads(2);rows=[];k=read(OLD/'inputs/continuous_lock.json')['kappa']
    for seed in 'ab':
        for sub in ('','confirm'):
            base=OLD/sub;bank=load(base/'records/text.pt');ext=HIST/'screen/records' if sub else base/'records'
            recs=[unpack(base/'records',seed),unpack(ext,'extra_'+seed)]
            tokens=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['native_tokens'].tolist()])
            for c in read(base/'inputs/panel.json'):
                nm=c['image'];vis=torch.cat([x[nm][1]['visual'] for x in recs]);proxy=sum((x[nm][1]['proxy_rank'] for x in recs),[])
                sc=row_scores(bank['native'].double().numpy(),bank['hard'].double().numpy(),tokens,bank['maps'][nm],vis,proxy,k)
                rows.append(dict(seed=seed,image=nm,pair=c['pair'],source=sub,scores=sc))
    dump(ROOT/'results/guard_train_choices.json',rows);dump(ROOT/'results/guard_train.json',summarize(rows))
if __name__=='__main__':main()
