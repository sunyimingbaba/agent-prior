"""Compute seed-paired contrasts with alternatives to the native input caption."""
import argparse
from pathlib import Path
import numpy as np
from comparisons import ARMS,SEEDS,ROOT,write
from diagnose import check_other_captions

def main(archive,arms):
    gallery=ROOT.parent/'E3-2_UCM双视图方位奖励_20261120/inputs'
    vectors=[];levels={}
    for sub in SEEDS:
        a,owner,valid=check_other_captions(gallery,archive/sub,arms,True)
        vectors.append(a);levels[sub]={arm:float(50*(i.mean()+t.mean())) for arm,(i,t) in a.items()}
    counts=np.bincount(owner)[valid]
    draws=np.random.default_rng(20261120).integers(len(valid),size=(20000,len(valid)))
    result={}
    for b in (arm for arm in arms if arm!='R2'):
        di=np.mean([v['R2'][0]-v[b][0] for v in vectors],axis=0)
        dt=np.mean([v['R2'][1]-v[b][1] for v in vectors],axis=0)
        summed=np.bincount(owner,weights=dt)[valid]
        boots=50*(di[draws].mean(1)+summed[draws].sum(1)/counts[draws].sum(1))
        lo,hi=np.quantile(boots,[.025,.975])
        result['R2-'+b]=dict(mean=float(50*(di.mean()+dt.mean())),lo=float(lo),hi=float(hi))
    write(ROOT/'results/other_captions_joint.json',dict(images=len(valid),captions=len(owner),levels=levels,contrasts=result))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('--primary',action='store_true')
    args=parser.parse_args()
    main(args.archive,['R0','R1','R2'] if args.primary else ARMS)
