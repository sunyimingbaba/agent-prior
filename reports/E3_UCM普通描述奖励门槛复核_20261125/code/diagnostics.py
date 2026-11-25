"""Descriptive checks on changed winners and conflicts between valid same-image captions."""
from collections import Counter
import gzip
import json
import numpy as np
from panels import ROOT,dump,read,question_panels,transfer_panel
from gate import BASE,I,J,rr_matrix

def main():
    rule=read(ROOT/'inputs/nomination.json')['rule'];allstats={}
    for tag in ('development','repeat','transfer'):
        primary='scene' if tag=='transfer' else 'question'
        counts=Counter();path=ROOT/'records'/(tag+'_folds.jsonl.gz')
        with gzip.open(path,'rt') as f:
            for line in f:
                row=json.loads(line)
                if row['mode']!=primary:continue
                a,b=[row['original'][key] for key in (BASE,rule)]
                counts['folds']+=1;counts['different_references']+=int(set(a['refs'])!=set(b['refs']))
                counts['different_gate']+=int(a['passed']!=b['passed']);counts['different_winners']+=int(a['winners']!=b['winners'])
                counts['improved_ordinary']+=int(b['mrr']>a['mrr']+1e-12)
                counts['worse_ordinary']+=int(b['mrr']<a['mrr']-1e-12)
                counts['improved_detail']+=int(b['detail']>a['detail']+1e-12)
                counts['worse_detail']+=int(b['detail']<a['detail']-1e-12)
        raw=transfer_panel() if tag=='transfer' else question_panels(tag)
        frac=[];pairs=conflicting=0;total=0
        for c,_,mode,b,v,_,_,_,_ in raw:
            if mode!=primary:continue
            nm=c['image'];pos=b['maps'][nm]['native'];alts=sorted(set(pos)-{pos[0]})
            mask=~b['even'].copy();mask[pos]=False
            sim=v[:16]@b['native'].T;rr=rr_matrix(sim[:,alts],sim[:,mask]).astype(float)
            diff=rr[I]-rr[J];any_signal=np.any(diff!=0,axis=1);both=(diff>0).any(1)&(diff<0).any(1)
            cnt=int(any_signal.sum());pairs+=cnt;conflicting+=int(both.sum());total+=1
            if cnt:frac.append(float(both.sum()/cnt))
        allstats[tag]=dict(changes=dict(counts),comparable_candidate_pairs=pairs,opposing_pairs=conflicting,
            groups=total,groups_with_comparisons=len(frac),mean_fraction_conflicting=float(np.mean(frac)))
    dump(ROOT/'results/diagnostics.json',allstats)
if __name__=='__main__':main()
