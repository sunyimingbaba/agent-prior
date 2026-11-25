"""A single preregistered vote override, including an additional disjoint 122-image panel."""
import sys
from fractions import Fraction
from collections import defaultdict
import numpy as np
import torch
from panels import ROOT,PRIOR,REP,RFPO,QDIR,read,load,dump,bank,question_panels,transfer_panel
from gate import wins,K,rr_matrix
from run_stage import boot
from verify import winners

def extra():
    hist=REP/'E3-2_UCM有效图组定向短训_20261124'
    sys.path.insert(0,str(hist/'code'))
    from verify_screen import unpack
    path=PRIOR/'guard';rows=read(path/'inputs/panel.json');b=bank(path)
    prior=read(PRIOR/'reward_confirm/inputs/panel.json')
    for key,other in [('image','image'),('rgb_sha256','rgb')]:
        used={c[key] for c in prior}|{c[other] for q in QDIR.values() for c in read(q/'inputs/lock.json')['rows']}
        assert len(used)==250 and not used.intersection(c[key] for c in rows)
    assert len(rows)==122
    for seed in 'ab':
        packs=[unpack(path/'records',tag) for tag in (seed,'extra_'+seed)]
        anchor=load(hist/'pilot/vectors'/(seed+'_start_fresh.pt'))
        assert anchor['images']==[r['image'] for r in rows]
        for i,c in enumerate(rows):
            nm=c['image'];rec=[p[nm] for p in packs];assert all(z['adapter']==anchor['adapter'] for z,_ in rec)
            v=torch.cat([z['visual'] for _,z in rec]+[anchor['features'][i:i+1]]).double().numpy()
            ranks=np.array(rec[0][1]['proxy_rank']+rec[1][1]['proxy_rank'])
            yield c,seed,'scene',b,v,ranks,'CDE',None,None

def one(c,b,v,ranks,letters):
    ix=b['maps'][c['image']];p0=ix['native'][0]
    pos=[p0]+sorted(set(ix['native'])-{p0},key=b['hashes'].__getitem__)
    ns=v@b['native'].T;hs=v@b['hard'].T
    use=np.ones(len(b['even']),bool);use[pos]=False
    left=ns[:,use&b['even']];right=ns[:,use&~b['even']]
    tr=rr_matrix(ns[:,pos],left);te=rr_matrix(ns[:,pos],right)
    m=ns[:,p0]-left.max(1);d=hs[:,ix['A'][0]]-hs[:,ix['A'][1]]
    detail=np.mean([hs[:,ix[l][0]]>hs[:,ix[l][1]] for l in letters],axis=0)
    r2=wins(m[:16],ranks)
    dense=np.unique((m+K*d)[:16],return_inverse=True)[1]
    floors=tr[:16][r2].sum(0)/int(r2.sum())
    legacy=tr[:16,:2].sum(1)>=floors[:2].sum()
    guard=wins(dense+17*legacy,ranks)
    legacy_mrr=te[:,2:].astype(float).max(1)
    historical={name:[detail[:16][mask].mean(),legacy_mrr[:16][mask].mean()] for name,mask in [('R2',r2),('guard',guard)]}
    rows=[]
    for fold in range(1,len(pos)):
        cols=[i for i in range(len(pos)) if i!=fold]
        values=tr[:16,cols];threshold=floors[cols]
        baseline=values[:,:2].sum(1)>=threshold[:2].sum()
        plus=(values>threshold).sum(1);minus=(values<threshold).sum(1);half=len(cols)/2
        patch=np.where(plus>half,True,np.where(minus>half,False,baseline))
        target=te[:,fold];true=target[:16]>=sum(target[:16][r2],Fraction())/int(r2.sum())
        good=(detail[:16]>detail[16])&(target[:16]>target[16]);changed=baseline!=patch
        row=dict(opportunity=float(good.any()),changed=float(changed.mean()),corrected=float(np.mean(changed&(patch==true))),
            error_added=float(np.mean(changed&(patch!=true))))
        for name,mask in [('baseline',baseline),('vote',patch)]:
            win=wins(dense+17*mask,ranks)
            assert np.flatnonzero(win).tolist()==winners(dense+17*mask,ranks)
            row[name]=dict(mrr=float(target[:16][win].mean()),detail=float(detail[:16][win].mean()),joint=float(good[win].mean()))
        rows.append(row)
    out={k:np.mean([r[k] for r in rows]) for k in ('opportunity','changed','corrected','error_added')}
    for name in ('baseline','vote'):
        out[name]={k:np.mean([r[name][k] for r in rows]) for k in ('mrr','detail','joint')}
    return out,historical

def main():
    torch.set_num_threads(2);summary={}
    for tag in ('development','repeat','transfer','additional122'):
        raw=extra() if tag=='additional122' else transfer_panel() if tag=='transfer' else question_panels(tag)
        mode='scene' if tag in ('transfer','additional122') else 'question';rows=[];prev=defaultdict(list)
        for c,s,md,b,v,rank,letters,_,_ in raw:
            if md!=mode:continue
            out,h=one(c,b,v,rank,letters);rows.append(dict(image=c['image'],seed=s,pair=c['pair'],**out))
            for key,x in h.items():prev[key].append(x)
        labels=[r['pair'] for r in rows];metrics={}
        def calc(a,b):
            z=boot(a,b,labels)
            z['seeds']={seed:sum(x for x,r in zip(a,rows) if r['seed']==seed)/sum(x for x,r in zip(b,rows) if r['seed']==seed) for seed in 'ab'}
            return z
        for key in ('mrr','detail','joint','joint_if_available'):
            field='joint' if key=='joint_if_available' else key
            den=np.array([r['opportunity'] if key=='joint_if_available' else 1. for r in rows])
            a,b=[np.array([r[m][field] for r in rows]) for m in ('baseline','vote')]
            metrics[key]=dict(baseline=calc(a,den),vote=calc(b,den),change=calc(b-a,den))
        if tag=='additional122':
            older=read(RFPO/'results/reward_replay.json')['fresh']['levels_exact']
            for key,x in prev.items():assert np.allclose(np.mean(x,axis=0),older[key],atol=1e-10,rtol=0)
        else:
            report=read(ROOT/'results'/(tag+'.json'))['levels']['hash_two']
            for key,data in metrics.items():assert abs(data['baseline']['mean']-report[key]['mean'])<1e-10
        changed=np.array([r['changed'] for r in rows]);ones=np.ones(len(rows))
        summary[tag]=dict(images=len(rows)//2,metrics=metrics,gate_changed=calc(changed,ones),
            corrected_given_change=calc(np.array([r['corrected'] for r in rows]),changed))
        dump(ROOT/'records'/('vote_'+tag+'.json'),rows)
    dump(ROOT/'results/vote.json',summary)
if __name__=='__main__':main()
