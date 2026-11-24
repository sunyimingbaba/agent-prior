import sys,hashlib,numpy as np,torch
from scipy.stats import rankdata
from candidate_guard import ROOT,HIST,OLD,read,load,dump,calibrated,summarize,unpack
def digest(x):return hashlib.sha256(','.join(map(str,x)).encode()).hexdigest()
def reciprocal(x,y):
    return np.array([np.mean(1/np.arange(1+(n>p).sum(),2+(n>p).sum()+(n==p).sum())) for p,n in zip(x,y)])
def analyze(tag):
    torch.set_num_threads(2);k=read(OLD/'inputs/continuous_lock.json')['kappa'];rows=[];diagnostics=[]
    for seed in 'ab':
        for sub in (('','confirm') if tag=='train' else (('reward_confirm',) if tag=='independent' else ('guard',))):
            base=ROOT/sub if sub in ('guard','reward_confirm') else OLD/sub;bank=load(base/'records/text.pt')
            extra=HIST/'screen/records' if sub=='confirm' else base/'records';recs=[unpack(base/'records',seed),unpack(extra,'extra_'+seed)]
            hashes=[digest(v) for v in bank['native_tokens'].tolist()];even=np.array([int(v[-1],16)%2==0 for v in hashes])
            nat=bank['native'].double().numpy();h=bank['hard'].double().numpy()
            for c in read(base/'inputs/panel.json'):
                nm=c['image'];v=torch.cat([r[nm][1]['visual'] for r in recs]).double().numpy();pr=np.array(sum((r[nm][1]['proxy_rank'] for r in recs),[]));ix=bank['maps'][nm];p=ix['native']
                alt=sorted(set(p)-{p[0]},key=lambda i:hashes[i]);trained=[p[0]]+alt[:1] if len(alt)>=2 else [p[0]];held=alt[1:] if len(alt)>=2 else alt;assert held
                use=np.ones(len(even),dtype=bool);use[p]=False;sim=v@nat.T;hs=v@h.T;x=sim[:,use&even];y=sim[:,use&~even]
                native=sim[:,p[0]]-x.max(1);a=hs[:,ix['A'][0]]-hs[:,ix['A'][1]];orig=native+k*a;proxy=1/(1+pr)
                def winners(order):
                    rw=calibrated(order,proxy);return np.flatnonzero(rw==rw.max())
                r2=winners(native);ref=np.mean([reciprocal(sim[:,i],x) for i in trained],axis=0);limit=ref[r2].mean()
                ok=ref+1e-14>=limit;new=rankdata(orig,method='dense')+(1+len(native))*ok
                rr=reciprocal(sim[:,held].max(1),y);sc={}
                for name,ind in [('R2',r2),('R3',winners(orig)),('guard',winners(new))]:
                    keys='CDE' if sub in ('guard','reward_confirm') else ('BCD' if sub=='confirm' else 'BC')
                    sc[name]=dict(rr=float(rr[ind].mean()),**{l:float((hs[ind,ix[l][0]]>hs[ind,ix[l][1]]).mean()) for l in keys})
                rows.append(dict(image=nm,pair=c['pair'],seed=seed,scores=sc));diagnostics.append((len(trained),np.mean(~ok[winners(new)])))
    dump(ROOT/'results'/('multi_'+tag+'.json'),dict(comparisons=summarize(rows),auxiliary_available=sum(n==2 for n,_ in diagnostics),groups=len(rows),winner_below_reference=float(np.mean([v for n,v in diagnostics]))));dump(ROOT/'results'/('multi_'+tag+'_rows.json'),rows)
if __name__=='__main__':
    for v in sys.argv[1:] or ('train','fresh'):analyze(v)
