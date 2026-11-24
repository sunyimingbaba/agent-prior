import sys
import hashlib
import numpy as np
from report_io import ROOT,LAST,HIST,OLD,read,dump,load,sha
from orders import score
sys.path.insert(0,str(LAST/'code'))
from multi_caption import digest,reciprocal
from candidate_guard import unpack

def main():
    folder=LAST/'reward_confirm';bank=load(folder/'records/text.pt');meta=read(folder/'inputs/panel.json')
    hashes=[digest(z) for z in bank['native_tokens'].tolist()];even=np.array([int(x,16)%2==0 for x in hashes])
    native=bank['native'].double().numpy();hard=bank['hard'].double().numpy()
    k=read(OLD/'inputs/continuous_lock.json')['kappa'];cases={};extras=[]
    for seed in 'ab':
        recs=[unpack(folder/'records',s) for s in (seed,'extra_'+seed)];rs=[]
        for c in meta:
            n=c['image'];g=[r[n] for r in recs];aa=[z[0] for z in g];assert aa[0]['first']==aa[1]['first']
            if not aa[0]['post_tool']:continue
            choices=aa[0]['second']+aa[1]['second']
            vis=np.concatenate([z[1]['visual'].double().numpy() for z in g]);rank=sum((z[1]['proxy_rank'] for z in g),[])
            s,h=vis@native.T,vis@hard.T;ix=bank['maps'][n];p=ix['native'];alt=sorted(set(p)-{p[0]},key=hashes.__getitem__)
            train=[p[0]]+alt[:1] if len(alt)>1 else [p[0]];held=alt[1:] if len(alt)>1 else alt
            mask=~np.isin(np.arange(len(even)),p);x=s[:,mask&even];a=h[:,ix['A'][0]]-h[:,ix['A'][1]]
            keep=np.mean([reciprocal(s[:,j],x) for j in train],axis=0);nr=s[:,p[0]]-x.max(1)
            rw,_=score([dict(rank=pr,keep=r,native=v,A=z) for pr,r,v,z in zip(rank,keep,nr,a)],k);u,v=rw['R2'],rw['R3']
            # Resolve duplicate slots and ties without consulting any held-out value.
            pairs=[]
            for i,good in enumerate(choices):
                for j,other in enumerate(choices):
                    gap=min(v[i]-v[j],u[j]-u[i])
                    if gap<=1e-12 or good==other:continue
                    div=next((t for t,(a,b) in enumerate(zip(good,other)) if a!=b),None)
                    if div is None:continue
                    tie=hashlib.sha256((str(good)+'|'+str(other)).encode()).hexdigest()
                    pairs.append((-gap,tie,i,j,div))
            if not pairs:continue
            gap,_,i,j,div=min(pairs)
            rs.append(dict(image=n,pair=c['pair'],first=aa[0]['first'],tool=aa[0]['tool_calls'][0],common=choices[i][:div],target=choices[i][div],alternative=choices[j][div],r3_sequence=choices[i],r2_sequence=choices[j],r3_gap=float(v[i]-v[j]),r2_gap=float(u[j]-u[i])))
            votes=np.mean(np.stack([h[:,ix[q][0]]>h[:,ix[q][1]] for q in 'CDE']),axis=0)
            rr=reciprocal(s[:,held].max(1),s[:,mask&~even]);extras.append(dict(seed=seed,image=n,detail_gap=float(votes[i]-votes[j]),rr_gap=float(rr[i]-rr[j])))
        cases[seed]=rs
    dump(ROOT/'inputs/probe_pairs.json',cases);dump(ROOT/'results/probe_pair_readouts.json',extras)
    dump(ROOT/'inputs/probe_lock.json',dict(cases=sha(ROOT/'inputs/probe_pairs.json'),protocol=sha(ROOT/'回答偏好复核口径.md'),counts={s:len(v) for s,v in cases.items()}))
if __name__=='__main__':main()
