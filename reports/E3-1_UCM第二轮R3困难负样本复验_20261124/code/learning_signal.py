"""Distinguish a change of top-set under uniform tie breaking from an actual policy-gradient signal."""
import csv,json,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from analyze import ROOT,OUT,write_csv,dump
from scipy.stats import rankdata
def reward(order,rank):
    # Literal order-statistic mapping in previous train.py.
    ranked=np.argsort(order,kind='stable');result=np.empty(len(rank));values=np.sort(1/(1+rank))
    for m in np.unique(order):
        ix=np.where(order[ranked]==m)[0];result[ranked[ix]]=values[ix].mean()
    return result
def advantage(score):
    v=(score-score.mean())/(score.std(ddof=1)+1e-8)
    return np.where(abs(v)>1e-10,v,0.)
def main(path):
    groups=defaultdict(list)
    with (path/'candidates.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):groups[r['arm'],r['image']].append(r)
    summary=defaultdict(Counter);rows=[]
    for (arm,name),rr in groups.items():
        m,a,rank=([float(r[k]) for r in rr] for k in ('native_margin','A','proxy_rank'))
        m,a,rank=map(np.array,(m,a,rank))
        base=reward(m,rank);old=advantage(base);best=set(np.flatnonzero(base==base.max()))
        for key,w in [('R3',.25),('R3_50',.5)]:
            score=reward((1-w)*rankdata(m)+w*rankdata(a),rank)
            adv=advantage(score);win=set(np.flatnonzero(score==score.max()))
            flag=dict(changed_winners=best!=win,expanded_tie=best<win,disjoint_winners=best.isdisjoint(win),
                old_flat=not np.any(old),new_flat=not np.any(adv),active_to_flat=bool(np.any(old)) and not np.any(adv),
                changed_advantage=bool(np.max(abs(adv-old))>1e-5),full_advantage_reversal=float(adv@old)<-1e-8)
            vals={}
            for k in ('B','C','D'):
                if k not in rr[0]:continue
                truth=np.array([float(r[k])>0 for r in rr])
                delta=float(truth[list(win)].mean()-truth[list(best)].mean())
                vals[k+'_change']=delta
                vals[k+'_change_when_active']=delta if np.any(adv) else 0.
            rows.append(dict(arm=arm,image=name,reward=key,**{k:int(v) for k,v in flag.items()},**vals))
            count=summary[arm+'_'+key];count.update({k:int(v) for k,v in flag.items()});count['images']+=1
            for k,v in vals.items():count[k]+=v
    write_csv(path/'learning_signal.csv',rows)
    dump(path/'learning_signal.json',{k:dict(v) for k,v in summary.items()})
if __name__=='__main__': main(Path(sys.argv[1]))
