"""Evaluate one preregistered panel. A committed development nomination is required for confirmation."""
import gzip
import io
import sys
import json
from collections import Counter,defaultdict
import numpy as np
import torch
from panels import ROOT,QDIR,PRIOR,read,dump,sha,question_panels,transfer_panel
from gate import BASE,ALTERNATIVES,NAMES,prepare,evaluate

def mean(values):
    valid=[x for x in values if x is not None]
    return float(np.mean(valid)) if valid else None

def aggregate(folds):
    cols=('concordance','mrr','detail','joint','opportunity','blocked','false_accept','false_reject')
    return {rule:{k:mean([f[rule][k] for f in folds]) for k in cols} for rule in NAMES}

def boot(n,d,labels):
    n=np.asarray(n,float);d=np.asarray(d,float);valid=np.isfinite(n)&np.isfinite(d)
    labels=list(labels);unique=list(dict.fromkeys(labels));idx=np.array([unique.index(k) for k in labels])
    g=np.bincount(idx,weights=np.where(valid,n,0),minlength=len(unique))
    size=np.bincount(idx,weights=np.where(valid,d,0),minlength=len(unique))
    draw=np.random.default_rng(20261125).integers(len(unique),size=(20000,len(unique)))
    dn=size[draw].sum(1);ok=dn>0
    return dict(mean=float(g.sum()/size.sum()) if size.sum() else None,
        interval=np.quantile(g[draw].sum(1)[ok]/dn[ok],[.025,.975]).tolist() if ok.any() else [None,None],
        defined_images=int(valid.sum()),defined_draws=int(ok.sum()))

def comparison(rows,variant,left,right=None):
    # Within each photo average its two seeds first. Ratios for opportunities weight folds equally within an image.
    items=defaultdict(list)
    for r in rows:items[r['image']].append(r)
    result={}
    for metric in ('concordance','mrr','detail','joint','joint_if_available','all_good_blocked','opportunity','false_accept','false_reject'):
        ratio=metric in ('joint_if_available','all_good_blocked')
        field={'joint_if_available':'joint','all_good_blocked':'blocked'}.get(metric,metric)
        numer=[];den=[];labels=[];by_seed=defaultdict(list)
        for group in items.values():
            ns=[];ds=[]
            for row in group:
                a=row[variant][left];x=a[field]
                if x is None:continue
                d=a['opportunity'] if ratio else 1.
                val=x if right is None else x-row[variant][right][field]
                ns.append(val);ds.append(d)
                by_seed[row['seed']].append((val,d))
            numer.append(float(np.mean(ns)) if ns else np.nan);den.append(float(np.mean(ds)) if ds else np.nan)
            labels.append(group[0]['pair'])
        result[metric]=boot(numer,den,labels)
        result[metric]['seeds']={s:float(sum(x for x,d in a)/sum(d for x,d in a)) if sum(d for x,d in a) else None for s,a in by_seed.items()}
    return result

def main(tag):
    torch.set_num_threads(2)
    nomination=ROOT/'inputs/nomination.json'
    if tag!='development':
        pick=read(nomination);assert pick['protocol']==sha(ROOT/'实验口径.md') and pick['source']==sha(ROOT/'results/development.json')
        assert pick['code']=={n:sha(ROOT/'code'/n) for n in ('run_stage.py','panels.py','gate.py')}
    raw=transfer_panel() if tag=='transfer' else question_panels(tag)
    rows=[];audit=Counter();worst=0.;foldfile=ROOT/'records'/(tag+'_folds.jsonl.gz')
    with foldfile.open('wb') as fp, gzip.GzipFile(fileobj=fp,mode='wb',mtime=0,compresslevel=4) as stream, io.TextIOWrapper(stream,encoding='utf-8') as log:
        for c,seed,mode,b,v,rank,letters,old,check in raw:
            nm=c['image'];z=prepare(b,nm,v,rank,letters,old,check);worst=max(worst,z['error'])
            base=dict(image=nm,pair=c['pair'],seed=seed,mode=mode,attribute=c['attribute'])
            orig=[];flipped=[]
            for idx in range(1,len(z['pos'])):
                one=evaluate(z,idx);rev=evaluate(z,idx,1)
                orig.append(one);flipped.append(rev)
                log.write(json.dumps(dict(**base,held=z['pos'][idx],positive_ids=z['pos'],original=one,swapped=rev),allow_nan=False)+'\n')
                audit['folds']+=1
                audit['informative_folds']+=int(one[BASE]['concordance'] is not None)
            rows.append(dict(**base,rotation=aggregate(orig),fixed=aggregate([orig[-1]]),swapped=aggregate(flipped)))
            audit['pools']+=1;audit['candidate_positions']+=len(rank)
    dump(ROOT/'records'/(tag+'_images.json'),rows)
    main_mode='scene' if tag=='transfer' else 'question'
    selected_rows=[r for r in rows if r['mode']==main_mode]
    assert len(selected_rows)==(284 if tag=='transfer' else 108)
    levels={rule:comparison(selected_rows,'rotation',rule) for rule in NAMES}
    nominated=(max(ALTERNATIVES,key=lambda r:levels[r]['concordance']['mean']) if tag=='development' else pick['rule'])
    deltas={}
    for mode in dict.fromkeys(r['mode'] for r in rows):
        grp=[r for r in rows if r['mode']==mode]
        deltas[mode]={v:comparison(grp,v,nominated,BASE) for v in ('rotation','fixed','swapped')}
    out=ROOT/'results'/(tag+'.json')
    dump(out,dict(nominated=nominated,main_mode=main_mode,levels=levels,delta=deltas,coverage=dict(audit),
        historical_max_error=worst,fold_sha=sha(foldfile),images_sha=sha(ROOT/'records'/(tag+'_images.json'))))
    if tag=='development':
        data=dict(rule=nominated,source=sha(out),protocol=sha(ROOT/'实验口径.md'),
            code={n:sha(ROOT/'code'/n) for n in ('run_stage.py','panels.py','gate.py')})
        if nomination.exists():assert read(nomination)==data
        else:dump(nomination,data)
if __name__=='__main__':main(sys.argv[1])
