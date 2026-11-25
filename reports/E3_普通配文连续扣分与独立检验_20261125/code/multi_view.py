"""A two-view detail score and ordinary estimates from a disjoint UCM caption donor bank."""
from collections import defaultdict
from functools import lru_cache
import hashlib, itertools, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from compare import OUT,SOURCE,OLD,PRIOR,read,sha,dump,load,bank,rr,K,calibrate,boot,METRICS,reduce,summary
from sheets import candidates
PROTO=OUT/'双视图与普通配文邻居口径.md';LOCK=OUT/'inputs/multi_nomination.json'
GRID=[(a,b,c) for a,b,c in itertools.product((1,2,4),(.5,1,2),(0,1))]
def name(w):return 'mix_'+'_'.join(map(str,w))

@lru_cache(maxsize=1)
def donor_bank():
    t,idx,allowed=candidates();remove={r['image'] for r in read(OUT/'inputs/fresh.json')+read(OUT/'inputs/extension.json')}
    ids=read(OUT/'inputs/native_index.json')['columns'];by=defaultdict(list);names=set()
    for n in sorted(set(allowed)-remove):
        cols=sorted({ids[s] for s in t['caption'][idx[n]].as_py()})
        by[(int(n[:-4])-1)//100].append((n,cols));names.add(n)
    return by,names

def native_context(b):
    by,names=donor_bank();t=b['native'];ctx={}
    for sid,donors in by.items():
        proto=np.stack([t[ids].mean(0) for _,ids in donors]);proto/=np.linalg.norm(proto,axis=1,keepdims=True)
        ctx[sid]=(proto,donors)
    return ctx,names

def neighbors(ctx,visible,pos,t):
    center=t[visible].mean(0);proto,ds=ctx
    top=np.argsort(-(proto@center),kind='stable')[:12];exclude=set(pos);freq=defaultdict(float)
    for i in top:
        use=[j for j in ds[i][1] if j not in exclude]
        for j in use:freq[j]+=1/len(use)
    if not freq:return [],[]
    ix=list(freq);w=np.array([freq[j] for j in ix]);return ix,w/w.sum()

def signature():
    return {f.name:sha(f) for f in [PROTO,OUT/'code/multi_view.py',OUT/'code/compare.py',OUT/'code/sheets.py',OUT/'inputs/native_index.json',OUT/'inputs/sampling_lock.json']}

def candidate_features(sim,ranks,neg,h,v,neigh_sim,neigh_rank):
    marg=sim-neg[:,None];hard=h.mean(1);rrmean=ranks.mean(1)
    basic=np.column_stack([sim,marg,ranks,marg.mean(1),marg.min(1),ranks.min(1),rrmean,
                           neigh_sim,neigh_sim-neg,neigh_rank,h,hard,h.min(1),h.max(1),
                           (h>0).astype(float),np.mean(h>0,axis=1),np.tanh(50*h),np.tanh(50*hard)])
    change=basic-basic[-1];dist=np.linalg.norm(v-v[-1],axis=1)
    hard_gain=hard-hard[-1];native_gain=rrmean-rrmean[-1];ng=neigh_rank-neigh_rank[-1]
    inter=np.column_stack([dist,native_gain*hard_gain,ng*hard_gain,(native_gain>0)&(hard_gain>0),
                           (ng>0)&(hard_gain>0),np.minimum(native_gain,np.tanh(50*hard_gain)),
                           (native_gain>0)&(hard>0),np.minimum(ng,np.tanh(50*hard_gain))])
    return np.column_stack([basic,change,inter])[:16]

def folds(tag):
    fresh=tag in ('fresh','extension');src=OUT if fresh else (OLD if tag=='first' else PRIOR/('guard' if tag=='third' else 'reward_confirm'))
    b=bank(src);ctx,donors=native_context(b)
    panel=read((OUT if fresh else SOURCE)/'inputs'/(tag+'.json'));assert not donors.intersection(r['image'] for r in panel)
    if not fresh:
        ref={(r['image'],r['seed']):r['free' if tag=='third' else 'scene'] for r in read(SOURCE/'records'/('controlled_timing_third.json' if tag=='third' else tag+'_images.json'))}
    output=[];missing=[]
    for seed in 'ab':
        if fresh:
            base=OUT/'records'/tag/seed/'features'
            sources={n:v for j in range(2) for n,v in read(base/('complete_'+str(j)+'.json'))['files'].items()}
            assert set(sources)=={r['image'] for r in panel}
        for c in panel:
            nm=c['image'];ix=b['maps'][nm];p0=ix['native'][0];pos=[p0]+sorted(set(ix['native'])-{p0},key=b['hashes'].__getitem__)
            if len(pos)<3:
                if seed=='a':missing.append(nm)
                continue
            if fresh:
                p=base/(nm+'.pt');assert sha(p)==sources[nm];z=load(p);assert z['rgb']==c['rgb_sha256'];v=z['visual'].double().numpy()
            else:
                p=(SOURCE/'records/prefix/third'/seed if tag=='third' else SOURCE/'records'/tag/seed)/'features'/(nm+'.pt')
                blob=load(p);assert blob['rgb']==c['rgb_sha256'];modes=blob['modes'];z=modes['free' if tag=='third' else 'scene'];v=z['visual'].double().numpy()
                if tag=='third':v[-1]=modes['locked']['visual'][-1].double().numpy()
            rk=np.asarray(z['ranks'][:16]);ns=v@b['native'].T;hs=v@b['hard'].T
            keep=np.ones(len(b['even']),bool);keep[pos]=False;left=ns[:,keep&b['even']];right=ns[:,keep&~b['even']]
            nmax=left.max(1);native=ns[:,p0]-nmax
            h=np.column_stack([hs[:,ix[k][0]]-hs[:,ix[k][1]] for k in 'AB'])
            detail=np.mean([hs[:,ix[k][0]]>hs[:,ix[k][1]] for k in 'CDE'],axis=0)
            dense=[np.unique((native+K*hh)[:16],return_inverse=True)[1] for hh in (h[:,0],h.mean(1))]
            r2=calibrate(native[:16],rk);r2w=np.flatnonzero(r2==r2.max())
            tr=np.array([[rr(s[t],n) for t in pos] for s,n in zip(ns,left)],object)
            val=np.array([[rr(s[t],n) for t in pos] for s,n in zip(ns,right)],object)
            # Search sorted training negatives only, used for scores of donor captions.
            sorted_n=np.sort(left,axis=1)
            for j in range(1,len(pos)):
                refs=[k for k in range(len(pos)) if k!=j][:2];vis=[pos[k] for k in refs]
                q=tr[:,refs].mean(1);floor=sum(q[r2w])/len(r2w);passed=q[:16]>=floor
                nn,wt=neighbors(ctx[c['scene_id']],vis,pos,b['native'])
                if len(nn):
                    sim=ns[:,nn];rank=1/(1+np.stack([len(n)-np.searchsorted(n,x,side='right') for n,x in zip(sorted_n,sim)]))
                    nm_sim=sim@wt;nm_rank=rank@wt
                else:nm_sim=ns[:,vis].mean(1);nm_rank=q.astype(float)
                x=candidate_features(ns[:,vis],tr[:,refs].astype(float),nmax,h,v,nm_sim,nm_rank)
                y=np.column_stack([(val[:16,j]>val[-1,j])&(detail[:16]>detail[-1]),val[:16,j].astype(float),detail[:16]])
                identity=tag+'|'+nm
                output.append(dict(image=nm,tag=tag,pair=c['pair'],seed=seed,id=identity,attr=c['attribute'],
                    weight=1/(2*16*(len(pos)-1)),x=x,y=y,rk=rk,anchor=np.array([0,float(val[-1,j]),detail[-1]]),
                    r2=r2,old=calibrate(dense[0]+17*passed,rk),ab=calibrate(dense[1]+17*passed,rk),
                    ref=None if fresh else ref[nm,seed],held=pos[j]))
    return output,missing

def fit(fs):
    x=np.vstack([f['x']-f['x'].mean(0) for f in fs]);y=np.vstack([f['y']-f['y'].mean(0) for f in fs])
    w=np.repeat([f['weight'] for f in fs],16);std=StandardScaler(with_mean=False).fit(x,sample_weight=w)
    model=Ridge(alpha=1,fit_intercept=False,solver='svd').fit(std.transform(x),y,sample_weight=w)
    return dict(scale=std.scale_.tolist(),coef=model.coef_.tolist())

def pred(f,model):
    return (f['x']/np.asarray(model['scale']))@np.asarray(model['coef']).T

def collect(fs,grid,models,extra=True):
    rows=defaultdict(lambda:defaultdict(list));metadata={}
    for f in fs:
        key=(f['tag'],f['image'],f['seed']);b=rows[key];metadata[key]=f
        yy=f['y'];b['opportunity'].append(yy[:,0].max());b['anchor'].append(f['anchor']);b['random'].append(yy.mean(0))
        scores={k:f[k] for k in ('r2','old','ab')}
        if grid:
            z=pred(f,models.get(f['pair'],models.get('all')))
            for w in grid:scores[name(w)]=calibrate(z@np.array([w[2],w[0],w[1]]),f['rk'])
        for k,r in scores.items():b['R2' if k=='r2' else k].append(yy[r==r.max()].mean(0))
    result=[]
    for key,b in rows.items():
        f=metadata[key];v={k:np.asarray(z).mean(0).tolist() for k,z in b.items()}
        if f['ref'] is not None:assert np.allclose(v['old'],[f['ref'][k] for k in METRICS],atol=1e-12,rtol=0)
        result.append(dict(tag=f['tag'],image=f['image'],pair=f['pair'],seed=f['seed'],attribute=f['attr'],means=v))
    return result

def develop():
    fs=[f for t in ('first','second','third') for f in folds(t)[0]]
    ps=sorted({f['pair'] for f in fs},key=lambda p:hashlib.sha256(('three-task|'+p).encode()).digest());g={p:k%5 for k,p in enumerate(ps)}
    models={}
    for k in range(5):
        model=fit([f for f in fs if g[f['pair']]!=k]);models.update({p:model for p in ps if g[p]==k})
    rs=collect(fs,GRID,models);allm=['ab']+[name(w) for w in GRID]
    s=summary(rs,['random',*allm],False);a=s['methods']
    def diff(m,t):return a[m][t]['change']['mean']
    chosen=[];guard=None
    for guard in (0.,-.01):
        chosen=[w for w in GRID if diff(name(w),'mrr')>=guard and diff(name(w),'detail')>=0]
        if chosen:break
    w=max(chosen,key=lambda w:diff(name(w),'joint')) if chosen else None
    m=name(w) if w else 'ab'
    dump(OUT/'records/multi_development.json',rs);dump(OUT/'results/multi_development.json',s)
    dump(LOCK,dict(method=m,weights=w,model=fit(fs),files=signature(),folds=g,guard=guard,donors=len(donor_bank()[1]),development=sha(OUT/'results/multi_development.json')))
    print(m,guard,[(k,round(diff(m,k),6)) for k in METRICS],flush=True)

def contrasts(rs,method,base):
    n=len(rs);labs=[r['tag']+'|'+r['pair'] for r in rs];a=np.array([np.array(r['means'][method])-r['means'][base] for r in rs])
    return {k:boot(a[:,j],np.ones(n),labs) for j,k in enumerate(METRICS)}
def confirm():
    l=read(LOCK);assert l['files']==signature() and l['development']==sha(OUT/'results/multi_development.json')
    grid=[l['weights']] if l['weights'] else [];m=l['method'];allrows=[];ss={};excluded={}
    for tag in ('extension','fresh'):
        fs,excluded[tag]=folds(tag);rows=collect(fs,grid,{'all':l['model']});allrows.extend(rows)
        dump(OUT/'records'/('multi_'+tag+'.json'),rows);ss[tag]=summary(rows,['random','ab']+([m] if grid else []))
    ss['both']=summary(allrows,['random','ab']+([m] if grid else []))
    for tag in ss:
        rr=[r for r in allrows if tag=='both' or r['tag']==tag]
        ss[tag]['contrasts']={ref:contrasts(rr,m,ref) for ref in ('old','R2','anchor','random')}
    d=ss['both']['contrasts']['old']
    eligible=all(ss[t]['contrasts']['old']['joint']['mean']>0 for t in ('extension','fresh')) and d['joint']['interval'][0]>0 and d['mrr']['interval'][0]>-.01 and d['detail']['mean']>=0
    dump(OUT/'results/multi_confirmation.json',dict(nomination=sha(LOCK),eligible=eligible,excluded=excluded,**ss))
    print(m,eligible,d,flush=True)
if __name__=='__main__':{'develop':develop,'confirm':confirm}[sys.argv[1]]()
