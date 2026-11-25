"""Replay the exact deployed reward on the previous 168-image archive and compare winner outcomes."""
import sys,numpy as np
from report_io import ROOT,LAST,HIST,OLD,read,dump
from orders import score
sys.path.insert(0,str(LAST/'code'))
from multi_caption import reciprocal,digest
from candidate_guard import unpack,load,calibrated
from scipy.stats import rankdata

def run():
    truth={(r['image'],r['seed']):r['scores'] for r in read(LAST/'results/multi_train_rows.json')}
    k=read(OLD/'inputs/continuous_lock.json')['kappa']
    groups=0;max_error=0.;winner_checks=0
    for seed in 'ab':
        for sub in ('','confirm'):
            folder=OLD/sub;bank=load(folder/'records/text.pt')
            records=[unpack(folder/'records',seed),unpack(HIST/'screen/records' if sub else folder/'records','extra_'+seed)]
            hashes=[digest(z) for z in bank['native_tokens'].tolist()]
            even=np.array([int(h[-1],16)%2==0 for h in hashes])
            native,hard=[bank[name].double().numpy() for name in ('native','hard')]
            for c in read(folder/'inputs/panel.json'):
                n=c['image'];vs=np.concatenate([r[n][1]['visual'].double().numpy() for r in records])
                ranks=sum((r[n][1]['proxy_rank'] for r in records),[]);ix=bank['maps'][n]
                p=ix['native'];alts=sorted(set(p)-{p[0]},key=hashes.__getitem__)
                train=[p[0]]+alts[:1] if len(alts)>1 else [p[0]];held=alts[1:] if len(alts)>1 else alts
                neg=~np.isin(np.arange(len(even)),p);s=vs@native.T;h=vs@hard.T;x=s[:,neg&even];y=s[:,neg&~even]
                margin=s[:,p[0]]-x.max(1);a=h[:,ix['A'][0]]-h[:,ix['A'][1]]
                keep=np.mean([reciprocal(s[:,j],x) for j in train],axis=0)
                gain=reciprocal(s[:,held].max(1),y)
                rows=[dict(native=m,A=v,keep=f,rank=q) for m,v,f,q in zip(margin,a,keep,ranks)]
                rew,floor=score(rows,k);proxy=1/(1+np.array(ranks));orig=calibrated(margin,proxy)
                limit=keep[orig==orig.max()].mean()
                comps=(orig,calibrated(rankdata(margin+k*a,method='dense')+17*(keep+1e-14>=limit),proxy))
                for name,ref,external in zip(('R2','R3'),comps,('R2','guard')):
                    assert np.array_equal(rew[name],ref) and abs(floor-limit)<1e-14
                    win=rew[name]==rew[name].max();obs=truth[n,seed][external]
                    vals={'rr':gain[win].mean(),**{l:(h[win,ix[l][0]]>h[win,ix[l][1]]).mean() for l in obs if l!='rr'}}
                    max_error=max(max_error,max(abs(vals[j]-obs[j]) for j in vals));winner_checks+=1
                groups+=1
    assert groups==336 and max_error<1e-10
    dump(ROOT/'results/reward_replay.json',dict(image_seed_groups=groups,candidate_positions=groups*16,winner_checks=winner_checks,max_abs_error=max_error))
if __name__=='__main__':run()
