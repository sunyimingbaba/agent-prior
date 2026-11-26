"""Measure whether one of the 17 actually observed trajectories reaches every target margin on a training view."""
import numpy as np
import torch
from local import ROOT,load,read,dump,feature_path

def main():
    traces=read(ROOT/'inputs/teacher_trace.json');plan=read(ROOT/'inputs/train.json')
    out=[]
    for s in 'ab':
        target=load(ROOT/'inputs'/('targets_'+s+'.pt'));rows={r['image']:r for r in plan[s]}
        for t in (t for t in traces if t['seed']==s):
            n=t['image'];k=t['view'];z=load(feature_path(rows[n],s,k))['visual'].double();g=target[n,k]
            ordinary=torch.relu(g['ordinary_target'].double()[None]-z@g['ordinary_dir'].double().T)
            hard=torch.relu(g['teacher'].double()[None]-z@g['hard_dir'].double().T)
            gap=torch.maximum(ordinary.max(1).values,hard.max(1).values);oracle=int(gap.argmin())
            w=t['winner'];out.append(dict(image=n,seed=s,view=k,winner=w,min_max_shortage=float(gap[oracle]),closest=oracle,
                teacher_max_shortage=float(gap[w]),hard_reachable=bool((hard.max(1).values<=1e-6).any()),
                ordinary_reachable=bool((ordinary.max(1).values<=1e-6).any()),both_reachable=bool((gap<=1e-6).any())))
    subset=[z for z in out if z['winner']!=16];other=[z for z in out if z['winner']==16]
    assert all(z['both_reachable'] for z in other)
    dump(ROOT/'results/feasibility.json',dict(eligible=len(subset),both=sum(z['both_reachable'] for z in subset),
        hard=sum(z['hard_reachable'] for z in subset),ordinary=sum(z['ordinary_reachable'] for z in subset),
        median_min_shortage=float(np.median([z['min_max_shortage'] for z in subset])),rows=out))
if __name__=='__main__':main()
