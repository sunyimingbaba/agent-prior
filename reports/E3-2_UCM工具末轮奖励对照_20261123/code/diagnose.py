"""Describe what changed the policy update, and recheck recall on captions distinct from the input."""
from collections import Counter
import numpy as np
import torch
from shared import ROOT, REBUILD, read, dump, setup
from metrics import fixed, recalls, ci

def reward_diagnostics():
    results={}
    for arm in ['R1','R2']:
        steps=read(ROOT/arm/'steps.json')
        stats=Counter()
        sizes=[]
        for s in steps:
            old=1/(1+np.asarray(s['proxy_rank']))
            native=np.array(s['native_margin']) if arm=='R2' else old
            fmt=np.array(s['format'])
            total=np.array(s['total'])
            best=total==total.max()
            stats['groups']+=1
            stats['eos_only_trajectories']+=sum(len(t['ids'])==1 and t['ended'] for t in s['trajectories'])
            stats['trajectories']+=len(s['trajectories'])
            stats['valid_answer_trajectories']+=sum(f>0 for f in s['format'])
            stats['winner_excludes_all_best_retrieval']+=int(not np.any(best & (native==native.max())))
            stats['winner_sets_collapsed_by_scale']+=int(np.sum(np.array(s['retrieval'])==max(s['retrieval']))>sum(native==native.max()))
            if arm=='R2':
                old_total=old+fmt
                stats['different_best_vs_proxy_with_same_format']+=int(not np.array_equal(best,old_total==old_total.max()))
                stats['different_update_vs_proxy']+=int(not np.allclose(total-total.mean(),old_total-old_total.mean(),atol=1e-12))
            sizes.extend(len(t['ids']) for t in s['trajectories'])
        stats.update(dict(shortest=min(sizes),longest=max(sizes)))
        results[arm]=dict(stats)
    return results

def check_other_captions():
    inputs=REBUILD/'paired_views_20261120/inputs'
    bank=torch.load(inputs/'retrieval_text.pt',map_location='cpu',weights_only=True)
    order=np.asarray(bank['caption_order']).reshape(-1,5)
    n=len(order)
    # Keep genuine alternative captions. Each image can contribute a different number of captions.
    idx=np.flatnonzero((order!=order[:,:1]).ravel())
    owners=idx//5
    valid=np.unique(owners)
    labels=np.arange(n)[:,None]==owners[None,:]
    byarm={}
    for arm in ['R0','R1','R2']:
        v=torch.stack([torch.load(ROOT/arm/'eval'/f'{j:03d}.pt',weights_only=True,map_location='cpu')['feature'] for j in range(n)])
        sim=fixed(v,bank['native'])[:,order.ravel()[idx]]
        # Also discard for each I2T source any candidate reproducing its own input caption.
        i2t_sim=sim.copy()
        i2t_sim[order[:,0,None]==order.ravel()[idx][None,:]]=-np.inf
        img=recalls(i2t_sim[valid],labels[valid])
        txt=recalls(sim.T,labels.T)
        byarm[arm]=(sum(img.values())/3,np.mean(list(txt.values()),axis=0))
    results={}
    for k,(a,b) in byarm.items():
        results[k]=dict(I2T_percent=100*float(a.mean()),T2I_percent=100*float(b.mean()),
                        mR_percent=50*float(a.mean()+b.mean()))
    diffs={}
    for a,b in [('R2','R1'),('R2','R0'),('R1','R0')]:
        di=byarm[a][0]-byarm[b][0]
        dt=byarm[a][1]-byarm[b][1]
        sums=np.bincount(owners,weights=dt,minlength=n)[valid]
        counts=np.bincount(owners,minlength=n)[valid]
        draws=np.random.default_rng(20261120).integers(len(valid),size=(20000,len(valid)))
        boot=50*(di[draws].mean(1)+sums[draws].sum(1)/counts[draws].sum(1))
        lo,hi=np.quantile(boot,[.025,.975])
        diffs[a+'-'+b]=dict(mean=float(50*(di.mean()+dt.mean())),lo=float(lo),hi=float(hi))
    return dict(images=len(valid),captions=len(idx),image_gallery=n,metrics=results,differences_pp=diffs)


def deployment_changes():
    vectors={}
    for arm in ['R0','R1','R2']:
        outputs=[torch.load(ROOT/arm/'eval'/f'{j:03d}.pt',map_location='cpu',weights_only=True) for j in range(210)]
        vectors[arm]=(torch.stack([x['prior'].float() for x in outputs]),torch.stack([x['feature'].float() for x in outputs]),
                      [[r['ids'] for r in x['rounds']] for x in outputs])
    results={}
    for a,b in [('R2','R1'),('R2','R0'),('R1','R0')]:
        ha,fa,ta=vectors[a]
        hb,fb,tb=vectors[b]
        results[a+'-'+b]=dict(identical_token_traces=sum(x==y for x,y in zip(ta,tb)),
            identical_priors=int((ha==hb).all(dim=1).sum()),mean_prior_cosine=float(torch.nn.functional.cosine_similarity(ha,hb).mean()),
            mean_image_feature_cosine=float(torch.nn.functional.cosine_similarity(fa,fb).mean()))
    return results

if __name__=='__main__':
    setup()
    out=ROOT/'results'
    dump(out/'diagnostics.json',dict(training=reward_diagnostics(),other_captions=check_other_captions(),deployment=deployment_changes()))
