"""Measure cross-seed, cross-prompt transfer of opportunities for actual RFPO training."""
import numpy as np
from compare import OUT,bank,read,dump,load,rr,sha
from exact_reward import exact_rewards

def events(v,b,c):
    n=c['image'];ix=b['maps'][n]
    positives=ix['native'];alts=sorted(set(positives)-{positives[0]})
    assert alts
    mask=~b['even'].copy();mask[positives]=False
    scores=v@b['native'].T; hs=v@b['hard'].T
    ds=sum((hs[:,ix[k][0]]>hs[:,ix[k][1]]).astype(int) for k in 'AB')
    rrs=[[rr(row[j],row[mask]) for j in alts] for row in scores]
    sc={'detail':ds.tolist(),'rr':[[[int(f.numerator),int(f.denominator)] for f in row] for row in rrs]}
    rew,joint=exact_rewards(sc)
    a,h=rew['ordinary'],rew['joint']
    return dict(opportunity=bool(max(joint)),changed=not np.array_equal(a,h),winner=not np.array_equal(a==a.max(),h==h.max()))

def ci(values):
    return np.quantile(values,[.025,.975]).tolist() if len(values) else [None,None]

def summarize(rows):
    labels=[r['tag']+'|'+r['pair'] for r in rows]
    ids,cl=np.unique(labels,return_inverse=True)
    draws=np.random.default_rng(20261125).integers(len(ids),size=(20000,len(ids)))
    selected=np.array([r['marked'] for r in rows],float)
    def sums(x):return np.bincount(cl,weights=x,minlength=len(ids))[draws].sum(1)
    nn=sums(np.ones_like(selected));sn=sums(selected);valid=sn>0
    report=dict(images=len(rows)//2,groups=len(rows),marked=int(selected.sum()),coverage=float(selected.mean()),metrics={})
    for key in ('changed','opportunity','winner'):
        y=np.array([r['target'][key] for r in rows],float)
        sy=selected*y
        ys=sums(y);sys=sums(sy)
        delta=sys[valid]/sn[valid]-ys[valid]/nn[valid]
        report['metrics'][key]=dict(total=int(y.sum()),marked_total=int(sy.sum()),base=float(y.mean()),marked=float(sy.sum()/selected.sum()) if selected.sum() else None,
            increment=float(sy.sum()/selected.sum()-y.mean()) if selected.sum() else None,interval=ci(delta),
            captured=float(sy.sum()/y.sum()) if y.sum() else None)
    return report

def main():
    ro=read(OUT/'inputs/real_roster.json');events_by={};recs=[]
    for tag in ('fresh','extension','saved','extra'):
        if tag in ('fresh','extension'):b=bank(OUT)
        else:
            # bank() expects a records subfolder. The archived text pack already holds both banks.
            import hashlib
            z=load(OUT/'records/covered/bank/text.pt')
            hashes=[hashlib.sha256(','.join(map(str,x)).encode()).hexdigest() for x in z['native_tokens'].tolist()]
            b=dict(native=z['native'].double().numpy(),hard=z['hard'].double().numpy(),maps=z['maps'],even=np.array([int(h[-1],16)%2==0 for h in hashes]))
        rows=[r for r in ro if r['tag']==tag]
        for c in rows:
            n=c['image']
            for s in 'ab':
                scene=(OUT/'records'/tag/s/'features') if tag in ('fresh','extension') else (OUT/'records/covered'/tag/s/'features')
                for name,f in [('scene',scene/(n+'.pt')),('first',OUT/'records/real'/s/'features'/(n+'.pt'))]:
                    z=load(f);assert z['rgb']==c['rgb_sha256']
                    events_by[n,s,name]=events(z['visual'].double().numpy(),b,c)
            for s,t in [('a','b'),('b','a')]:
                src=events_by[n,s,'scene'];dst=events_by[n,t,'first']
                recs.append(dict(image=n,tag=tag,pair=c['pair'],from_seed=s,to_seed=t,marked=src['changed'],source=src,target=dst))
    summary={k:summarize([r for r in recs if r['tag'] in tags]) for k,tags in [('first138',('fresh','extension')),('other144',('saved','extra')),('all',('fresh','extension','saved','extra'))]}
    summary['eligible']=all(summary[k]['metrics']['changed']['increment']>0 for k in ('first138','other144')) and summary['all']['metrics']['changed']['interval'][0]>0
    summary['protocol']=sha(OUT/'有用训练图的跨种子核查.md')
    dump(OUT/'records/mining_images.json',recs);dump(OUT/'results/mining.json',summary)
if __name__=='__main__':main()
