"""Check the actual reward against historical vectors and quantify ties in the previous implementation."""
import sys,numpy as np
from scipy.stats import rankdata
from report_io import ROOT,LAST,HIST,OLD,read,dump
from orders import score
from exact_check import exact
sys.path.insert(0,str(LAST/'code'))
from multi_caption import reciprocal,digest
from candidate_guard import unpack,load,calibrated,boot

def run(tag):
    truth={(r['image'],r['seed']):r['scores'] for r in read(LAST/'results'/('multi_'+tag+'_rows.json'))}
    k=read(OLD/'inputs/continuous_lock.json')['kappa'];counts={};rows=[];err=0.;floating=0.;historical=0.
    for seed in 'ab':
        for base in ([OLD,OLD/'confirm'] if tag=='train' else [LAST/('guard' if tag=='fresh' else 'reward_confirm')]):
            ext=HIST/'screen/records' if base==OLD/'confirm' else base/'records'
            bank=load(base/'records/text.pt');recs=[unpack(base/'records',seed),unpack(ext,'extra_'+seed)]
            hashes=[digest(z) for z in bank['native_tokens'].tolist()]
            even=np.array([int(h[-1],16)%2==0 for h in hashes]);native,hard=[bank[n].double().numpy() for n in ('native','hard')]
            for c in read(base/'inputs/panel.json'):
                n=c['image'];vs=np.concatenate([r[n][1]['visual'].double().numpy() for r in recs]);rank=sum((r[n][1]['proxy_rank'] for r in recs),[]);ix=bank['maps'][n]
                p=ix['native'];alts=sorted(set(p)-{p[0]},key=hashes.__getitem__);train=[p[0]]+alts[:1] if len(alts)>1 else [p[0]];held=alts[1:] if len(alts)>1 else alts
                neg=~np.isin(np.arange(len(even)),p);s=vs@native.T;h=vs@hard.T;x=s[:,neg&even];y=s[:,neg&~even]
                margin=s[:,p[0]]-x.max(1);a=h[:,ix['A'][0]]-h[:,ix['A'][1]]
                keep=np.mean([reciprocal(s[:,j],x) for j in train],axis=0);gain=reciprocal(s[:,held].max(1),y)
                raw=[dict(native=m,A=v,keep=f,rank=q) for m,v,f,q in zip(margin,a,keep,rank)]
                rew,floor=score(raw,k);proxy=1/(1+np.array(rank));orig=calibrated(margin,proxy);dense=rankdata(margin+k*a,method='dense')
                r2=exact(margin,rank);limit=keep[r2==r2.max()].mean();assert abs(limit-floor)<1e-14
                order=dense+17*(keep+1e-14>=limit)
                old_ord=dense+17*(keep+1e-14>=keep[orig==orig.max()].mean())
                res={};pref=('R2','R3','guard')
                for name,new,old in [('R2',r2,orig),('R3',exact(margin+k*a,rank),calibrated(margin+k*a,proxy)),('guard',exact(order,rank),calibrated(old_ord,proxy))]:
                    if name!='R3':err=max(err,float(np.max(np.abs(new-rew['R2' if name=='R2' else 'R3']))))
                    before=old==old.max();after=new==new.max()
                    if not np.array_equal(before,after):counts[name]=counts.get(name,0)+1
                    cmp=calibrated(order,proxy) if name=='guard' else old
                    floating=max(floating,float(np.max(np.abs(new-cmp))))
                    obs=truth[n,seed][name]
                    for win,key in [(before,'before'),(after,'after')]:
                        vals={'rr':gain[win].mean(),**{q:(h[win,ix[q][0]]>h[win,ix[q][1]]).mean() for q in obs if q!='rr'}}
                        if key=='before':historical=max(historical,max(abs(vals[q]-obs[q]) for q in vals))
                        res[name+'_'+key]=[np.mean([v for q,v in vals.items() if q!='rr']),vals['rr']]
                rows.append(dict(seed=seed,pair=c['pair'],scores=res))
    assert err==0 and historical<1e-10 and floating<1e-14
    by={m:np.mean([r['scores'][m+'_after'] for r in rows],axis=0).tolist() for m in pref}
    changes={}
    for m in ('R3','guard'):
        z=np.array([[np.array(r['scores'][m+'_after'])-r['scores']['R2_after'] for r in rows if r['seed']==s] for s in 'ab'])
        changes[m+'-R2']={met:dict(**boot(z[:,:,j].mean(0),[r['pair'] for r in rows if r['seed']=='a']),seeds=z[:,:,j].mean(1).tolist()) for j,met in enumerate(['detail','rr'])}
    return dict(groups=len(rows),candidate_positions=len(rows)*16,observed_max_error=historical,exact_max_error=err,same_order_float_max_error=floating,winner_sets_changed=counts,levels_exact=by,differences_exact=changes)
if __name__=='__main__':dump(ROOT/'results/reward_replay.json',{tag:run(tag) for tag in ('train','fresh','independent')})
