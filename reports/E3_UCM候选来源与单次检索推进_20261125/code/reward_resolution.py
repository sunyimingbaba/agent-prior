"""Remove proxy payout ties, preserving the original ordinary threshold and hard-negative order."""
from collections import defaultdict
import sys
import numpy as np
from rosters import ROOT,REP,PRIOR,OLD,read,sha,dump
from report_scores import bank,load,rr
from orders import calibrate,advantages
from run_stage import boot
from gate import K

def per_image(c,b,feat):
    v=feat['visual'].double().numpy();ranks=np.asarray(feat['ranks'][:16]);ix=b['maps'][c['image']]
    pos=[ix['native'][0]]+sorted(set(ix['native'])-{ix['native'][0]},key=b['hashes'].__getitem__)
    mask=np.ones(len(b['even']),bool);mask[pos]=False
    ns=v@b['native'].T;hs=v@b['hard'].T;left=ns[:,mask&b['even']];right=ns[:,mask&~b['even']]
    margin=ns[:16,pos[0]]-left[:16].max(1);hard=hs[:16,ix['A'][0]]-hs[:16,ix['A'][1]]
    r2=calibrate(margin,ranks);winner=np.flatnonzero(r2==r2.max())
    dense=np.unique(margin+K*hard,return_inverse=True)[1]
    training=[[rr(ns[i,p],left[i]) for p in pos] for i in range(16)]
    validation=np.array([[rr(ns[i,p],right[i]) for p in pos] for i in range(17)],dtype=object)
    detail=np.mean([hs[:,ix[k][0]]>hs[:,ix[k][1]] for k in 'CDE'],axis=0)
    metrics=defaultdict(list);evidence=[]
    for j in range(1,len(pos)):
        refs=[k for k in range(len(pos)) if k!=j][:2]
        q=[sum(row[k] for k in refs) for row in training]
        floor=sum(q[i] for i in winner)/len(winner);passed=np.array([x>=floor for x in q]);order=dense+17*passed
        present=calibrate(order,ranks);ordinal=calibrate(order,np.arange(16))
        old=np.flatnonzero(present==present.max());new=np.flatnonzero(ordinal==ordinal.max())
        assert set(new)<=set(old) and set(new)==set(np.flatnonzero(order==order.max()))
        mrr=validation[:16,j].astype(float);good=((validation[:16,j]>validation[16,j])&(detail[:16]>detail[16])).astype(float)
        metrics['winners_pruned'].append(len(new)<len(old))
        metrics['distinct_winners_pruned'].append(len(np.unique(v[old],axis=0))>len(np.unique(v[new],axis=0)))
        for method,rew,ids in [('original',present,old),('ordinal',ordinal,new)]:
            adv=advantages(rew);metrics[method+'_constant'].append(np.all(rew==rew[0]))
            metrics[method+'_winners'].append(len(ids));metrics[method+'_distinct_winners'].append(len(np.unique(v[ids],axis=0)))
            for k,y in [('joint',good),('mrr',mrr),('detail',detail[:16])]:
                metrics[method+'_'+k].append(float(y[ids].mean()))
                metrics[method+'_signal_'+k].append(float(np.dot(adv,y-y.mean())/16))
        evidence.append(dict(held=pos[j],old=old.tolist(),new=new.tolist()))
    return {k:float(np.mean(z)) for k,z in metrics.items()},evidence

def main(tag):
    rows=read(ROOT/'inputs'/(tag+'.json'))
    b=bank(OLD if tag=='first' else PRIOR/('reward_confirm' if tag=='second' else 'guard'))
    reference=read(ROOT/'records'/((tag+'_images.json') if tag!='third' else 'controlled_timing_third.json'))
    verify={(c['image'],c['seed']):c['scene' if tag!='third' else 'free'] for c in reference}
    out=[]
    for s in 'ab':
        base=ROOT/'records'/tag/s if tag!='third' else ROOT/'records/prefix/third'/s
        for c in rows:
            nm=c['image']
            if (nm,s) not in verify:continue
            f=load(base/'features'/(nm+'.pt'));assert f['rgb']==c['rgb_sha256']
            z,fold=per_image(c,b,f['modes']['scene' if tag!='third' else 'free'])
            for k in ('joint','mrr','detail'):assert abs(z['original_'+k]-verify[nm,s][k])<1e-12
            out.append(dict(image=nm,seed=s,pair=c['pair'],values=z,folds=fold))
    stats={}
    for key in out[0]['values']:
        y=np.array([r['values'][key] for r in out]);stats[key]=boot(y,np.ones(len(y)),[r['pair'] for r in out])
        if key.startswith('ordinal_'):
            orig=key.replace('ordinal_','original_',1);old=np.array([r['values'][orig] for r in out])
            stats[key.replace('ordinal_','change_',1)]=boot(y-old,np.ones(len(y)),[r['pair'] for r in out])
    dump(ROOT/'results'/('resolution_'+tag+'.json'),dict(protocol=sha(ROOT/'奖励并列与训练信号复核口径.md'),
        images=len(out)//2,pairs=len({r['pair'] for r in out}),metrics=stats))
    dump(ROOT/'records'/('resolution_'+tag+'_images.json'),out)
def decision():
    vals={k:read(ROOT/'results'/('resolution_'+k+'.json'))['metrics'] for k in ('first','second','third')}
    eligible=all(x['change_joint']['mean']>0 for x in vals.values()) and vals['third']['change_joint']['interval'][0]>0
    eligible=eligible and all(x['change_'+k]['interval'][1]>=0 for x in vals.values() for k in ('mrr','detail'))
    dump(ROOT/'results/resolution_nomination.json',dict(eligible=bool(eligible),protocol=sha(ROOT/'奖励并列与训练信号复核口径.md'),sources={k:sha(ROOT/'results'/('resolution_'+k+'.json')) for k in vals}))
if __name__=='__main__':
    if sys.argv[1]=='decide':decision()
    else:main(sys.argv[1])
