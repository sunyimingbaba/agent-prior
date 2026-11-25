"""Develop one reward and compare its frozen predictions on disjoint image pairs."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from selection import features, model_order

OUT=Path(__file__).resolve().parents[1]
SOURCE=OUT.parent/'E3_UCM候选来源与单次检索推进_20261125'
sys.path.insert(0,str(SOURCE/'code'))
from reward_resolution import OLD,PRIOR,read,dump,sha,load,bank,rr,K,calibrate,boot
METRICS=('joint','mrr','detail')
RULES=('soft_0.5','soft_1','soft_2','soft_4','learned','protected')
LOCK=OUT/'inputs/nomination.json'
PROTO=OUT/'实验口径.md'

def panel(tag):
    roster=read(SOURCE/'inputs'/(tag+'.json'))
    src=OLD if tag=='first' else PRIOR/('guard' if tag=='third' else 'reward_confirm')
    b=bank(src)
    fname='controlled_timing_third' if tag=='third' else tag+'_images'
    audit={(r['image'],r['seed']):r['free' if tag=='third' else 'scene'] for r in read(SOURCE/'records'/(fname+'.json'))}
    folds=[]
    for seed in 'ab':
        base=SOURCE/'records'/tag/seed if tag!='third' else SOURCE/'records/prefix/third'/seed
        for row in roster:
            nm=row['image']
            if (nm,seed) not in audit:continue
            pack=load(base/'features'/(nm+'.pt'))
            assert pack['rgb']==row['rgb_sha256']
            modes=pack['modes'];z=modes['free' if tag=='third' else 'scene']
            v=z['visual'].double().numpy()
            if tag=='third':v[-1]=modes['locked']['visual'][-1].double().numpy()
            ix=b['maps'][nm];p0=ix['native'][0];pos=[p0]+sorted(set(ix['native'])-{p0},key=b['hashes'].__getitem__)
            mask=np.ones(len(b['even']),bool);mask[pos]=False
            ns=v@b['native'].T;hs=v@b['hard'].T
            left=ns[:,mask&b['even']];right=ns[:,mask&~b['even']]
            neg=left.max(1);margin=ns[:,p0]-neg;hard=hs[:,ix['A'][0]]-hs[:,ix['A'][1]]
            rank=np.asarray(z['ranks'][:16]);r2=calibrate(margin[:16],rank);top=np.flatnonzero(r2==r2.max())
            dense=np.unique((margin+K*hard)[:16],return_inverse=True)[1]
            train=np.array([[rr(s[p],n) for p in pos] for s,n in zip(ns,left)],object)
            held=np.array([[rr(s[p],n) for p in pos] for s,n in zip(ns,right)],object)
            detail=np.mean([hs[:,ix[k][0]]>hs[:,ix[k][1]] for k in 'CDE'],axis=0)
            for j in range(1,len(pos)):
                refs=[i for i in range(len(pos)) if i!=j][:2]
                q=train[:,refs].mean(1);floor=sum(q[top])/len(top);passed=q[:16]>=floor
                order=dense+17*passed
                joint=(held[:16,j]>held[16,j])&(detail[:16]>detail[16])
                actual=np.column_stack([joint,held[:16,j].astype(float),detail[:16]])
                f=dict(image=nm,seed=seed,pair=row['pair'],tag=tag,held=pos[j],weight=1/(2*(len(pos)-1)*16),
                       x=features(ns[:,[pos[i] for i in refs]],train[:,refs].astype(float),neg,hard,v),
                       y=actual,rank=rank,positive=joint.astype(int),passed=passed,
                       dense=dense,deficit=np.array([max(Fraction(),floor-i) for i in q[:16]],object),
                       anchor=np.array([0,float(held[16,j]),detail[16]]),r2=r2,
                       base=calibrate(order,rank),reference=audit[nm,seed])
                folds.append(f)
    return folds

def fit(fs):
    x=np.vstack([f['x'] for f in fs]);y=np.concatenate([f['positive'] for f in fs])
    w=np.repeat([f['weight'] for f in fs],16)
    # Preserve one total unit of evidence per photo in the penalized likelihood.
    std=StandardScaler().fit(x,sample_weight=w);z=std.transform(x)
    lr=LogisticRegression(C=1,max_iter=2000,tol=1e-9).fit(z,y,sample_weight=w)
    return dict(mean=std.mean_.tolist(),scale=std.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]))

def scores(f,rule,model):
    if rule=='old':return f['base']
    if rule=='R2':return f['r2']
    if rule.startswith('soft_'):
        l=Fraction(rule.split('_')[1]);d=f['dense'];m=max(int(d.max()),1)
        order=np.array([Fraction(int(i),m)-l*g for i,g in zip(d,f['deficit'])],object)
    else:
        order=model_order(f['x'],model)
        if rule=='protected':order=np.unique(order,return_inverse=True)[1]+17*f['passed']
    return calibrate(order,f['rank'])

def reduce(fs,methods,tag,models):
    buckets=defaultdict(lambda:defaultdict(list));refs={}
    for f in fs:
        key=(f['image'],f['seed'],f['pair']);b=buckets[key];refs[key]=f['reference']
        b['anchor'].append(f['anchor']);b['opportunity'].append(int(f['positive'].any()))
        for m in methods:
            r=scores(f,m,models.get(f['pair'],models.get('all')))
            win=np.flatnonzero(r==r.max());b[m].append(f['y'][win].mean(0))
    rows=[]
    for key,b in buckets.items():
        means={m:np.asarray(v).mean(0).tolist() for m,v in b.items()}
        assert np.max(abs(np.asarray(means['old'])-[refs[key][k] for k in METRICS]))<1e-12,key
        rows.append(dict(image=key[0],seed=key[1],pair=key[2],tag=tag,means=means))
    dump(OUT/'records'/(tag+'.json'),rows)
    return rows

def summary(rows,methods,interval=True):
    a={};labels=[r['tag']+'|'+r['pair'] for r in rows]
    for m in ['anchor','old','R2']+list(methods):
        vals=np.array([r['means'][m] for r in rows]);old=np.array([r['means']['old'] for r in rows])
        a[m]={}
        for j,k in enumerate(METRICS):
            diff=vals[:,j]-old[:,j]
            a[m][k]={'mean':float(vals[:,j].mean()),'change':(boot(diff,np.ones(len(rows)),labels) if interval else {'mean':float(diff.mean())}),
                     'seed_changes':{s:float(np.mean([d for d,r in zip(diff,rows) if r['seed']==s])) for s in 'ab'}}
    return dict(images=len({r['image'] for r in rows}),pairs=len(set(labels)),
                opportunity=float(np.mean([r['means']['opportunity'] for r in rows])),methods=a)

def hashes():
    return {p.name:sha(p) for p in [PROTO,OUT/'code/compare.py',OUT/'code/selection.py']}

def develop():
    fs=panel('first');pairs=sorted({f['pair'] for f in fs},key=lambda p:hashlib.sha256(('e3-oof|'+p).encode()).digest())
    by={p:k%5 for k,p in enumerate(pairs)};models={}
    for k in range(5):
        train=[f for f in fs if by[f['pair']]!=k];model=fit(train)
        models.update({p:model for p in pairs if by[p]==k})
    rows=reduce(fs,['old','R2',*RULES],'first',models);s=summary(rows,RULES)
    delta=lambda m,t:s['methods'][m][t]['change']['mean']
    choices=[m for m in RULES if delta(m,'mrr')>=-.01 and delta(m,'detail')>=0 and delta(m,'joint')>0]
    exploratory=not choices
    if not choices:choices=[m for m in RULES if delta(m,'mrr')>=-.01]
    pick=max(choices,key=lambda m:delta(m,'joint')) if choices else None
    dump(OUT/'results/development.json',s)
    dump(LOCK,dict(method=pick,exploratory=exploratory,model=fit(fs),folds=by,files=hashes(),development=sha(OUT/'results/development.json')))
    print(pick,exploratory,[(m,[round(delta(m,t),5) for t in METRICS]) for m in RULES],flush=True)

def confirm():
    lock=read(LOCK);assert lock['files']==hashes() and lock['development']==sha(OUT/'results/development.json')
    m=lock['method'];assert m
    combined=[];out={}
    seen={r['rgb_sha256'] for r in read(SOURCE/'inputs/first.json')}
    for tag in ('second','third'):
        roster=read(SOURCE/'inputs'/(tag+'.json'));rgb={r['rgb_sha256'] for r in roster};assert seen.isdisjoint(rgb);seen.update(rgb)
        fs=panel(tag);rows=reduce(fs,['old','R2',m],tag,{'all':lock['model']});out[tag]=summary(rows,[m]);combined.extend(rows)
    out['combined']=summary(combined,[m]);u=out['combined']['methods'][m]
    ok=all(out[t]['methods'][m]['joint']['change']['mean']>0 for t in ('second','third')) and u['joint']['change']['interval'][0]>0 and u['mrr']['change']['interval'][0]>-.01 and u['detail']['change']['mean']>=0
    dump(OUT/'results/confirmation.json',dict(nomination=sha(LOCK),eligible=ok,**out))
    print(m,ok,[(k,u[k]['change']) for k in METRICS],flush=True)

if __name__=='__main__':
    {'develop':develop,'confirm':confirm}[sys.argv[1]]()
