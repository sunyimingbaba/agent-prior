"""Take a training candidate only when it corrects the original greedy prior on the reward side."""
from collections import Counter
import sys
import numpy as np
import torch
from candidate_guard import ROOT, HIST, OLD, read, load, dump, unpack, boot
from multi_caption import digest
from recheck_reward import rr

def select(scores, allowed):
    if not np.any(allowed): return np.array([0])
    best = scores[allowed].max()
    return np.flatnonzero(allowed & (scores == best))

def study(tag):
    assert tag in ('fresh','independent')
    base = ROOT / ('reward_confirm' if tag=='independent' else 'guard')
    panel = read(base/'inputs/panel.json')
    bank = load(base/'records/text.pt')
    native, hard = [bank[k].double().numpy() for k in ('native','hard')]
    hashes = [digest(v) for v in bank['native_tokens'].tolist()]
    even = np.array([int(h,16)%2==0 for h in hashes])
    kappa = read(OLD/'inputs/continuous_lock.json')['kappa']
    rows=[]; counts={}; methods=('single','ordinary','correct')
    for seed in 'ab':
        archive = [unpack(base/'records',s) for s in (seed,'extra_'+seed)]
        anchor = load(ROOT/'pilot/vectors'/(seed+'_start_reward_anchor.pt') if tag=='independent' else HIST/'pilot/vectors'/(seed+'_start_fresh.pt'))
        assert anchor['images']==[r['image'] for r in panel]
        counts[seed]=dict(triggered=0,ordinary=0,correct=0,remaining_candidates=0)
        for i,c in enumerate(panel):
            name=c['image']; ix=bank['maps'][name]
            vs=torch.cat([anchor['features'][i:i+1]]+[a[name][1]['visual'] for a in archive]).double().numpy()
            assert vs.shape[0]==17
            s=vs@native.T; h=vs@hard.T
            p=ix['native'];alt=sorted(set(p)-{p[0]},key=hashes.__getitem__)
            train=[p[0]]+alt[:1] if len(alt)>1 else [p[0]]
            held=alt[1:] if len(alt)>1 else alt
            mask=~np.isin(np.arange(len(native)),p);neg=s[:,mask&even]
            training_rr=np.stack([rr(s[:,j],neg) for j in train],axis=1)
            margin=s[:,p[0]]-neg.max(axis=1)
            ab=np.stack([h[:,ix[k][0]]-h[:,ix[k][1]] for k in 'AB'],axis=1)
            trigger=bool(np.min(ab[0])<=0)
            eligible=(training_rr>=training_rr[0]-1e-12).all(axis=1)&(margin>=margin[0]-1e-12)&trigger
            eligible[0]=False
            fixes=(ab>=ab[0]-1e-12).all(axis=1)&((ab>0).sum(axis=1)>(ab[0]>0).sum())
            fix=eligible&fixes
            control=select(margin,eligible);pick=select(margin+kappa*ab[:,0],fix)
            n=counts[seed];n['triggered']+=int(trigger);n['ordinary']+=int(control[0]!=0);n['correct']+=int(pick[0]!=0);n['remaining_candidates']+=int(fix.sum())
            held_rank=rr(s[:,held].max(axis=1),s[:,mask&~even])
            votes=np.stack([h[:,ix[k][0]]>h[:,ix[k][1]] for k in 'CDE'],axis=1)
            sc={}
            for method,win in zip(methods,(np.array([0]),control,pick)):
                sc[method]=dict(detail=float(votes[win].mean()),rr=float(held_rank[win].mean()),improved=float((votes[win]&~votes[0]).sum(axis=1).mean()),lost=float((~votes[win]&votes[0]).sum(axis=1).mean()),chosen=win.tolist())
            rows.append(dict(seed=seed,image=name,pair=c['pair'],scores=sc))
    levels={m:{k:np.mean([r['scores'][m][k] for r in rows]) for k in ('detail','rr')} for m in methods}
    comparisons={}
    for a,b in [('correct','single'),('ordinary','single'),('correct','ordinary')]:
        stats={}
        for k in ('detail','rr'):
            x=np.array([[r['scores'][a][k]-r['scores'][b][k] for r in rows if r['seed']==seed] for seed in 'ab'])
            stats[k]=dict(**boot(x.mean(0),[r['pair'] for r in panel]),seeds=x.mean(axis=1).tolist())
        comparisons[a+'-'+b]=stats
    flips={m:{k:sum(r['scores'][m][k] for r in rows) for k in ('improved','lost')} for m in methods}
    dump(ROOT/'results'/('relative_'+tag+'_rows.json'),rows)
    dump(ROOT/'results'/('relative_'+tag+'.json'),dict(images=len(panel),levels=levels,counts=counts,expected_votes=flips,comparisons=comparisons))
if __name__=='__main__':study(sys.argv[1])
