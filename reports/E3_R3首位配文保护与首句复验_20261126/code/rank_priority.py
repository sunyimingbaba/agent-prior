"""Alternate ordering after the same two predicted gains and top-one constraint."""
from pathlib import Path
import gzip
import json
import numpy as np
import torch
from guard import (LAST,STAGE,ARMS,OLD,PRIOR,PRIOR_RUN,METRICS,best,choose,detailed,external,measure,
                   previous_banks,previous_panel,read,real_stream,with_gaps,summarize)
from 校准复验 import fit_model

ALL=ARMS+('rank_only','rank_priority')


def alt(packet,weights):
    selections,protected=choose(packet,weights)
    ref=packet['baseline'];gaps=packet['gaps']
    keep=(gaps[ref]>0).all(0)
    preds=packet['x']@weights
    gain=preds-preds[ref].mean(0)
    allowed=(gain>1e-12).all(1)&(gaps[:,keep]>0).all(1)
    order=list(zip(gain[:,1],gain[:,0],packet['ordinary']))
    selections['rank_priority']=best(order,np.flatnonzero(allowed)) or ref
    selections['rank_only']=best(list(zip(np.round(preds[:,1],12),packet['ordinary'])),range(8))
    return measure(packet,selections,protected)


def record(tag,rows):
    target=STAGE/'results'
    with gzip.open(target/('priority_'+tag+'.jsonl.gz'),'wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result=summarize(rows,ALL,('ordinary','calibrated','guard','rank_only'))
    for name in ('support_images',):result.pop(name)
    (target/('priority_'+tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def weight(model):
    return np.array(model['coef'])/np.array(model['scale_x'])[:,None]


def main(tag):
    archive=read(LAST/'锁定评分参数.json')
    rows=read(OLD/'inputs/real_roster.json')
    hold={r['rgb_sha256'] for r in read(PRIOR_RUN/'inputs/holdout.json')}
    w=weight(archive['model'])
    def evaluate(stream):return [alt(p,w) for p in with_gaps(stream)]
    if tag=='dev':
        stream=previous_panel([r for r in rows if r['rgb_sha256'] not in hold],previous_banks())
        packets=list(with_gaps(stream))
        checks=[]
        for fold in range(5):
            train=[p for p in packets if archive['folds'][p['pair']]!=fold]
            test=[p for p in packets if archive['folds'][p['pair']]==fold and p['view']=='first']
            model=fit_model(train)
            checks.extend(alt(p,weight(model)) for p in test)
        result=record(tag,checks)
        baseline=read(LAST/'校准结果.json')['dev_cv']['first']
        for a in ('ordinary','calibrated','rank_only'):
            for k in METRICS:assert abs(result['levels'][a][k]-baseline['levels'][a][k])<1e-12
    elif tag=='history':
        record('first78',evaluate(p for p in previous_panel([r for r in rows if r['rgb_sha256'] in hold],previous_banks()) if p[2]=='first'))
        pooled=[]
        for name,src in [('scene148',STAGE.parent/'E3_UCM多数配文新图复核_20261125'),('scene122',PRIOR/'guard')]:
            rec=evaluate(external(src,set()));record(name,rec);pooled.extend(rec)
        record('pooled_scene',pooled)
    else:
        record('first98',evaluate(real_stream()))


if __name__=='__main__':
    import sys
    torch.set_num_threads(1)
    for tag in sys.argv[1:]:main(tag)
