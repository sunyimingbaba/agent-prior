"""Descriptive ceiling from the 16 sampled priors, using held-out B/C only after sampling."""
import numpy as np
from report_io import ROOT,read,dump

cases=read(ROOT/'records/message_trial.json')['cases']
result={}
for mode in ('raw','chat'):
    varied=0
    winners_changed=0
    selected_detail_changed=0
    best=[]
    for row in cases.values():
        group=row[mode]
        votes=np.array([sum(s[key]>0 for key in ('B','C'))/2 for s in group['scores'][:-1]])
        assert len(votes)==16
        best.append(float(votes.max()))
        varied+=int(np.ptp(votes)>0)
        r2,r3=(np.asarray(group['retrieval'][reward]) for reward in ('R2','R3'))
        a,b=r2==r2.max(),r3==r3.max()
        winners_changed+=int(np.any(a!=b))
        selected_detail_changed+=int(abs(votes[a].mean()-votes[b].mean())>1e-12)
    result[mode]=dict(images=len(best),varying_BC_accuracy=varied,mean_oracle_best_BC=float(np.mean(best)),
        changed_winner_set=winners_changed,changed_winner_mean_BC=selected_detail_changed)
dump(ROOT/'results/message_headroom.json',result)
