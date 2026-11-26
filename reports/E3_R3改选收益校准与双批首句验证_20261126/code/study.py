"""Cross-fit candidate-level gain models on image pairs, then perform locked tests."""
from collections import defaultdict
import gzip
import itertools
import json
import sys
import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from data import ROOT, PREV, ARMS, read, sha, stream, packets, best, outcomes, summarize
from 校准复验 import fit_model as original_fit

FAMILIES=('ridge','trees')
WEIGHTS=(.5,1.,2.)
MODES=('rerank','positive','veto')
EPS=1e-10


def old_matrix(data):
    old_data=[]
    for p in data:
        target=p['y'][0][:,[1,2]]
        old_data.append(dict(x=p['old'][0],y=target-target.mean(0),weight=p['w']))
    cfg=original_fit(old_data)
    return np.array(cfg['coef'])/np.array(cfg['scale_x'])[:,None]


def fit(data):
    x=np.concatenate([p['x'].reshape(-1,p['x'].shape[-1]) for p in data]).astype(np.float32)
    y=np.concatenate([p['y'].reshape(-1,4) for p in data])
    w=np.repeat([p['w'] for p in data],24);w/=w.sum()
    sy=np.maximum(np.sqrt(w@(y*y)),1e-12)
    y=y/sy
    mx=w@x
    sx=np.maximum(np.sqrt(w@((x-mx)**2)),1e-9)
    normalized=(x-mx)/sx
    weighted=normalized.T*w
    coef=np.linalg.solve(weighted@normalized+.1*np.eye(x.shape[1]),weighted@y)
    ridge=dict(mean=mx,scale=sx,coef=coef)
    tree=ExtraTreesRegressor(n_estimators=72,max_depth=8,min_samples_leaf=80,
                              max_features=.7,n_jobs=4,random_state=261127)
    tree.fit(x,y,sample_weight=w*len(w))
    return dict(ridge=ridge,trees=tree,outcome_scale=sy)


def predictions(data,models,old_w,families=FAMILIES):
    x=np.stack([p['x'] for p in data]);shape=x.shape[:-1]+(4,);flat=x.reshape(-1,x.shape[-1])
    preds={}
    if 'ridge' in families:
        ridge=models['ridge']
        preds['ridge']=(((flat-ridge['mean'])/ridge['scale'])@ridge['coef']).reshape(shape)
    if 'trees' in families:
        preds['trees']=models['trees'].predict(flat).reshape(shape)
    old=np.stack([p['old']@old_w for p in data])
    return old,preds


def old_choices(pred,p):
    gains=pred-pred[p['ref']].mean(0)
    eligible=np.flatnonzero((gains>1e-12).all(1))
    return best(list(zip(gains.min(1),p['exact'])),eligible) or p['ref']


def choose(p,pred,old,weight,mode):
    gains=pred-pred[p['ref']].mean(0)
    fine=(gains[:,0]+gains[:,1])/2
    rank=gains[:,2]
    utility=fine+weight*rank+.5*gains[:,3]
    allowed=utility>EPS
    if mode=='positive':
        allowed &= (fine>EPS)&(rank>EPS)
    elif mode=='veto':
        return old if utility[old].mean()>EPS else p['ref']
    eligible=np.flatnonzero(allowed)
    return best(list(zip(np.round(utility,10),p['exact'])),eligible) or p['ref']


def dev_fit():
    data=list(packets(stream('dev')))
    lock=read(PREV/'锁定评分参数.json')
    pairs=lock['folds']
    assert {p['pair'] for p in data}==set(pairs)
    options=list(itertools.product(FAMILIES,WEIGHTS,MODES))
    totals=defaultdict(lambda:np.zeros(3))
    denom=0.
    for fold in range(5):
        training=[p for p in data if pairs[p['pair']]!=fold]
        val=[p for p in data if pairs[p['pair']]==fold and p['view']=='first']
        models=fit(training);old_w=old_matrix(training)
        old,pred=predictions(val,models,old_w)
        for i,p in enumerate(val):
            old_top=old_choices(old[i,0],p)
            labels=np.column_stack((p['c'],p['mrr']))
            old_level=labels[old_top].mean(0)
            delta=labels-labels[p['ref']].mean(0)
            totals['scale'][:2]+=p['w']*(delta**2).mean(0);denom+=p['w']
            for cfg in options:
                family,weight,mode=cfg
                ws=choose(p,pred[family][i,0],old_top,weight,mode)
                win=labels[ws].mean(0)-old_level
                totals[cfg][:2]+=p['w']*win
                totals[cfg][2]+=p['w']*(ws!=old_top)
        print('development fold',fold,flush=True)
    scales=np.sqrt(totals.pop('scale')[:2]/denom)
    def key(cfg):
        d=totals[cfg][:2]/denom
        return (bool((d>0).all()),min(d/scales),sum(d/scales))
    chosen=max(options,key=key)
    all_models=fit(data)
    path=ROOT/'inputs/switch_model.joblib';joblib.dump(all_models,path,compress=3)
    old_w=old_matrix(data)
    saved=np.array(lock['model']['coef'])/np.array(lock['model']['scale_x'])[:,None]
    assert np.allclose(saved,old_w,rtol=1e-8,atol=1e-8)
    cfg=dict(family=chosen[0],weight=chosen[1],mode=chosen[2],file_sha256=sha(path),old_model=sha(PREV/'锁定评分参数.json'),
             source={f.name:sha(f) for f in (ROOT/'code').glob('*.py')},development_images=204,outcome_scales=scales.tolist(),
             candidates=[dict(family=c[0],weight=c[1],mode=c[2],detail_delta=(totals[c]/denom)[0],
                              mrr_delta=(totals[c]/denom)[1],switch_rate=(totals[c]/denom)[2]) for c in sorted(options,key=key,reverse=True)])
    (ROOT/'inputs/提名.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print('nominated',cfg['candidates'][0],flush=True)


def evaluate(tag):
    cfg=read(ROOT/'inputs/提名.json')
    path=ROOT/'inputs/switch_model.joblib'
    assert sha(path)==cfg['file_sha256'] and sha(PREV/'锁定评分参数.json')==cfg['old_model']
    models=joblib.load(path)
    old_cfg=read(PREV/'锁定评分参数.json')['model']
    old_w=np.array(old_cfg['coef'])/np.array(old_cfg['scale_x'])[:,None]
    data=[p for p in packets(stream(tag)) if p['view']=='first']
    family=cfg['family']
    old,pred=predictions(data,models,old_w,(family,))
    rec=[]
    for i,p in enumerate(data):
        base=old_choices(old[i,0],p)
        triple=old_choices(old[i].mean(0),p)
        ranked=best(list(zip(np.round(old[i,0,:,1],12),p['exact'])),range(8))
        ab=choose(p,pred[family][i,0],base,cfg['weight'],cfg['mode'])
        full=choose(p,pred[family][i].mean(0),triple,cfg['weight'],cfg['mode'])
        rec.append(outcomes(p,dict(ordinary=p['ref'],old=base,old3=triple,switch_ab=ab,switch=full,rank_only=ranked)))
    save(tag,rec)
    if tag in ('history78','history98'):
        prior_folder=ROOT.parent/'E3_R3首位配文保护与首句复验_20261126'
        base=read(prior_folder/'results'/('first'+tag.removeprefix('history')+'.json'))
        now=read(ROOT/'results'/(tag+'.json'))
        assert now['images']==base['images'] and now['folds']==base['folds']
        for key in ('detail','mrr','net'):
            assert abs(now['levels']['old'][key]-base['levels']['calibrated'][key])<1e-10


def save(tag,rows):
    with gzip.open(ROOT/'results'/(tag+'.jsonl.gz'),'wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    stats=summarize(rows,ARMS,('ordinary','old','old3','rank_only','switch_ab'))
    stats.pop('support_images')
    (ROOT/'results'/(tag+'.json')).write_text(json.dumps(stats,indent=2)+'\n')
    print(tag,stats['images'],{k:round(v['mean'],6) for k,v in stats['against']['old']['switch'].items()},flush=True)


def merge():
    rows=[];used=set();pairids=set()
    for tag in ('v122','v148'):
        with gzip.open(ROOT/'results'/(tag+'.jsonl.gz'),'rt') as f:subset=[json.loads(s) for s in f]
        keys={r['rgb'] for r in subset};pairs={r['pair'] for r in subset}
        assert not used.intersection(keys) and not pairids.intersection(pairs)
        rows+=subset;used|=keys;pairids|=pairs
    save('pooled',rows)
    stats=read(ROOT/'results/pooled.json');matched=stats['against']['old']['switch'];base=stats['against']['ordinary']['switch']
    passed=[]
    for main,other in (('detail','mrr'),('mrr','detail')):
        passed.append(matched[main]['interval'][0]>0 and matched[other]['mean']>=0 and
                      all(read(ROOT/'results'/(t+'.json'))['against']['old']['switch'][main]['mean']>0 for t in ('v122','v148')))
    result={'improvement':bool(any(passed) and base['detail']['mean']>0 and base['mrr']['mean']>0),
            'detail_primary':passed[0],'mrr_primary':passed[1]}
    (ROOT/'results/首次判定.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    for tag in sys.argv[1:]:
        if tag=='fit':dev_fit()
        elif tag=='merge':merge()
        else:evaluate(tag)
