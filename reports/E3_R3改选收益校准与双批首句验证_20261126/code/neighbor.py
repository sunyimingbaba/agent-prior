"""Estimate the retrieval consequence of a switch from training-image caption prototypes."""
from collections import defaultdict
import gzip
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import torch
from data import ROOT, PREV, best, read, sha, stream, packets, outcomes, summarize
import importlib.util
spec=importlib.util.spec_from_file_location('switch_gain_study',Path(__file__).with_name('study.py'))
gain_study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gain_study)
old_matrix,old_choices=gain_study.old_matrix,gain_study.old_choices

SIZES=(0,8,32)
ORDERINGS=('both','sum')
ARMS=('ordinary','old','old3','local_visible','local_ab','local3','local_rank')
TINY=1e-10


def unit(v):
    return v/max(np.linalg.norm(v),1e-12)


def prototypes(raw):
    banks=defaultdict(list)
    for r,s,v,b,vis in raw:
        ids=sorted(set(b['maps'][r['image']]['native']))
        texts=b['native'][ids]
        banks[s,v].append((r['pair'],r['rgb_sha256'],unit(texts.mean(0)),vis[-1],texts))
    return banks


def quantiles_of_scores(scores,neg):
    sorted_neg=np.sort(neg,axis=1)
    return np.vstack([1/(1+len(pool)-np.searchsorted(pool,x,side='right')) for x,pool in zip(scores,sorted_neg)])


def stats_matrix(score,neg):
    pseudo=quantiles_of_scores(score,neg)
    maxneg=neg.max(1)
    eighth=np.partition(neg,-8,axis=1)[:,-8:].mean(1)
    return np.column_stack((score.mean(1),score.max(1),score.min(1),pseudo.mean(1),pseudo.max(1),
                            pseudo.min(1),score.mean(1)-maxneg,score.mean(1)-eighth))


def local_packets(raw,pool,exclude):
    for row,s,v,bank,visual in raw:
        if row['pair'] in exclude:continue
        ix=bank['maps'][row['image']]['native'];cols=sorted(set(ix)-{ix[0]})
        if len(cols)<2:continue
        library=[r for r in pool[s,v] if r[0]!=row['pair']]
        assert len(library)>=32
        texts=np.stack([r[2] for r in library])
        images=np.stack([r[3] for r in library])
        native=bank['native'];odd=~bank['even'].copy();odd[ix]=False
        group=list(packets([(row,s,v,bank,visual)]))
        for p in group:
            visible=[k for j,k in enumerate(cols) if j!=p['held']]
            textq=unit(native[[ix[0]]+visible].mean(0))
            sl=slice(8*p['block'],8*p['block']+8)
            vis=visual[sl]
            relative_image=images@unit(vis.mean(0))
            relative_text=texts@textq
            combined=relative_image/max(np.std(relative_image),1e-12)+relative_text/max(np.std(relative_text),1e-12)
            indices=np.argsort(-combined,kind='stable')
            neg=vis@native[odd].T
            g=vis@native[visible].T-neg.max(1,keepdims=True)
            base=np.column_stack((p['old'][0][:,6:],
                                  stats_matrix(vis@native[[ix[0]]].T,neg),
                                  stats_matrix(vis@native[visible].T,neg),
                                  stats_matrix((vis@textq)[:,None],neg),
                                  (g>0).mean(1),g.min(1)))
            features={}
            for k in SIZES:
                x=base
                if k:
                    picked=indices[:k]
                    support=np.concatenate([library[j][4] for j in picked])
                    center=unit(texts[picked].mean(0))
                    proxy=np.column_stack((stats_matrix(vis@support.T,neg),
                                           stats_matrix(vis@texts[picked].T,neg),
                                           stats_matrix((vis@center)[:,None],neg),
                                           (vis@(unit(textq+center)))[:,None]))
                    x=np.column_stack((base,proxy))
                delta=x-x[p['ref']].mean(0)
                # Interact slopes with the amount of room in the visible rankings.
                features[k]=np.column_stack((delta,delta*(1-p['ordinary'][p['ref']].mean())))
            yield dict(image=p['image'],pair=p['pair'],rgb=p['rgb'],seed=s,view=v,held=p['held'],block=p['block'],
                w=p['w'],ref=p['ref'],ordinary=p['ordinary'],exact=p['exact'],mrr=p['mrr'],c=p['c'],
                fine=p['fine'],old=p['old'],y=p['y'],local=features)


def rank_fit(rows,k):
    x=np.concatenate([p['local'][k] for p in rows])
    y=np.concatenate([p['mrr']-p['mrr'][p['ref']].mean() for p in rows])
    w=np.repeat([p['w'] for p in rows],8);w/=w.sum()
    sx=np.maximum(np.sqrt(w@(x*x)),1e-10);sy=max(np.sqrt(w@(y*y)),1e-12)
    norm=x/sx;wt=norm.T*w
    beta=np.linalg.solve(wt@norm+.1*np.eye(norm.shape[1]),wt@(y/sy))
    return beta/sx


def decide(p,old,rank,weight,ordering):
    diff=old[:,0]-old[p['ref'],0].mean()
    rank=rank-rank[p['ref']].mean()
    score=diff+weight*rank
    valid=score>TINY
    if ordering=='both':valid&=(diff>TINY)&(rank>TINY)
    return best(list(zip(np.round(score,10),p['exact'])),np.flatnonzero(valid)) or p['ref']


def coherent(pool,excluded):
    return {key:[v for v in values if v[0] not in excluded] for key,values in pool.items()}


def fit_dev():
    full=list(stream('dev'))
    data_by_pair=read(PREV/'锁定评分参数.json')['folds']
    pool=prototypes(full)
    settings=list(itertools.product(SIZES,(.5,1.,2.),ORDERINGS))
    totals={s:np.zeros(2) for s in settings};scales=np.zeros(2);den=0.
    for fold in range(5):
        hold={p for p,f in data_by_pair.items() if f==fold}
        eligible=coherent(pool,hold)
        train=list(local_packets((q for q in full if q[0]['pair'] not in hold),eligible,set()))
        val=list(local_packets((q for q in full if q[0]['pair'] in hold and q[2]=='first'),eligible,set()))
        coefs={k:rank_fit(train,k) for k in SIZES};oldw=old_matrix(train)
        for p in val:
            op=p['old'][0]@oldw;start=old_choices(op,p);y=np.column_stack((p['c'],p['mrr']))
            ref=y[start].mean(0);d=y-y[p['ref']].mean(0)
            den+=p['w'];scales+=p['w']*(d*d).mean(0)
            for k in SIZES:
                ranks=p['local'][k]@coefs[k]
                for weight,ordering in itertools.product((.5,1.,2.),ORDERINGS):
                    top=decide(p,op,ranks,weight,ordering)
                    totals[k,weight,ordering]+=p['w']*(y[top].mean(0)-ref)
        print('caption neighborhood fold',fold,flush=True)
        del train,val
    scales=np.sqrt(scales/den)
    def key(cfg):
        change=totals[cfg]/den
        return (bool((change>0).all()),min(change/scales),sum(change/scales))
    options=sorted(settings,key=key,reverse=True)
    # A separate no-neighbor ablation retains the selected weights and action.
    winner=options[0]
    all_rows=list(local_packets(full,pool,set()));model={k:rank_fit(all_rows,k).tolist() for k in SIZES}
    chosen=dict(k=winner[0],weight=winner[1],mode=winner[2],coefficients=model,
                development=[dict(k=a,weight=b,mode=c,changes=(totals[a,b,c]/den).tolist()) for a,b,c in options],
                source=sha(Path(__file__)),old=sha(PREV/'锁定评分参数.json'))
    (ROOT/'inputs/配文预测提名.json').write_text(json.dumps(chosen,indent=2)+'\n')
    print(chosen['development'][0],flush=True)


def run(tag):
    cfg=read(ROOT/'inputs/配文预测提名.json')
    assert sha(Path(__file__))==cfg['source'] and cfg['old']==sha(PREV/'锁定评分参数.json')
    old=read(PREV/'锁定评分参数.json')['model']
    oldw=np.array(old['coef'])/np.array(old['scale_x'])[:,None]
    pool=prototypes(stream('dev'))
    rows=[]
    for p in local_packets((q for q in stream(tag) if q[2]=='first'),pool,set()):
        choice={}
        local_rank=p['local'][cfg['k']]@np.array(cfg['coefficients'][str(cfg['k'])])
        no_neighbor=p['local'][0]@np.array(cfg['coefficients']['0'])
        old_all=p['old']@oldw;old3=old_all.mean(0);ab=old_all[0]
        choice.update(ordinary=p['ref'],old=old_choices(ab,p),old3=old_choices(old3,p),
              local_ab=decide(p,ab,local_rank,cfg['weight'],cfg['mode']),
              local3=decide(p,old3,local_rank,cfg['weight'],cfg['mode']),
              local_visible=decide(p,old3,no_neighbor,cfg['weight'],cfg['mode']),
              local_rank=best(list(zip(np.round(local_rank,10),p['exact'])),range(8)))
        rows.append(outcomes(p,choice))
    save(tag,rows)


def save(tag,rows):
    dest=ROOT/'results'
    with gzip.open(dest/('neighbor_'+tag+'.jsonl.gz'),'wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    stats=summarize(rows,ARMS,('ordinary','old','old3','local_visible','local_rank','local_ab'))
    stats.pop('support_images')
    if tag=='pooled':
        grouped=defaultdict(list)
        for r in rows:grouped[r['pair'],r['image'],r['seed']].append(r)
        pairs=sorted({p for p,_,_ in grouped});rng=np.random.default_rng(261126)
        counts=rng.multinomial(len(pairs),np.ones(len(pairs))/len(pairs),size=20000)
        n=np.array([sum(p==q for p,_,_ in grouped) for q in pairs]);denom=counts@n
        for k in ('detail','mrr','net'):
            bypair=np.array([sum(np.mean([r['outcomes']['local3'][k]-r['outcomes']['old'][k] for r in g]) for (p,_,_),g in grouped.items() if p==q) for q in pairs])
            stats['against']['old']['local3'][k]['interval_975']=np.quantile((counts@bypair)/denom,[.0125,.9875]).tolist()
    (dest/('neighbor_'+tag+'.json')).write_text(json.dumps(stats,indent=2)+'\n')
    print(tag,stats['images'],{k:round(v['mean'],5) for k,v in stats['against']['old']['local3'].items()},flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    for tag in sys.argv[1:]:
        if tag=='fit':fit_dev()
        elif tag=='merge':
            rows=[]
            for t in ('v122','v148'):
                with gzip.open(ROOT/'results'/('neighbor_'+t+'.jsonl.gz'),'rt') as f:rows.extend(json.loads(s) for s in f)
            save('pooled',rows)
        else:run(tag)
