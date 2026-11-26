"""Decompose an actual changed decision into agreeing held detail and caption movements."""
from collections import defaultdict
import gzip
import json
import numpy as np
import torch
from data import ROOT,read,stream,packets

PAIRS=(('old','ordinary'),('old3','old'),('switch','old3'),('switch','old'),('local3','old3'),('local3','old'))


def logs(tag,prefix):
    with gzip.open(ROOT/'results'/(prefix+tag+'.jsonl.gz'),'rt') as f:
        return {(r['image'],r['seed'],r['block'],r['held']):r for r in map(json.loads,f)}


def main():
    per_image=defaultdict(list)
    overlap=set()
    for tag in ('v122','v148'):
        primary=logs(tag,'');secondary=logs(tag,'neighbor_')
        assert set(primary)==set(secondary)
        for p in packets(stream(tag)):
            key=(p['image'],p['seed'],p['block'],p['held'])
            saved=primary[key]
            assert saved['rgb']==p['rgb'] and saved['pair']==p['pair']
            tops={a:q['top'] for a,q in saved['outcomes'].items()}
            tops['local3']=secondary[key]['outcomes']['local3']['top']
            for a,b in PAIRS:
                target=tops[a];ref=tops[b];changed=target!=ref
                df=p['fine'][target,None]-p['fine'][ref]
                dr=p['mrr'][target,None]-p['mrr'][ref]
                winner=float(((df>0)&(dr>0)).mean()) if changed else 0.
                loss=float(((df<0)&(dr<0)).mean()) if changed else 0.
                plus=float(((df>=0)&(dr>=0)&((df>0)|(dr>0))).mean()) if changed else 0.
                minus=float(((df<=0)&(dr<=0)&((df<0)|(dr<0))).mean()) if changed else 0.
                per_image[tag,p['rgb'],p['seed'],a,b].append((changed,winner,loss,plus,minus,float(df.mean()),float(dr.mean())))
            overlap.add((tag,p['rgb']))
    out={}
    for tag in ('v122','v148','pooled'):
        out[tag]={}
        for a,b in PAIRS:
            arrays=[np.array(v) for (t,_,_,x,y),v in per_image.items() if (tag=='pooled' or t==tag) and (x,y)==(a,b)]
            means=np.mean([v.mean(0) for v in arrays],axis=0)
            count=np.concatenate(arrays)[:,0].sum()
            out[tag][a+'-'+b]=dict(folds=sum(len(v) for v in arrays),changes=int(count),changed_rate=means[0],
                    win_given_change=float(means[1]/means[0]) if count else None,loss_given_change=float(means[2]/means[0]) if count else None,
                    pareto_plus_given_change=float(means[3]/means[0]) if count else None,pareto_minus_given_change=float(means[4]/means[0]) if count else None,
                    detail=means[5],mrr=means[6])
            source=read(ROOT/'results'/(('neighbor_' if a=='local3' else '')+tag+'.json'))
            for j,k in [(5,'detail'),(6,'mrr')]:
                diff=source['levels'][a][k]-source['levels'][b][k]
                assert abs(means[j]-diff)<1e-12
    (ROOT/'results/改选逐项核对.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
