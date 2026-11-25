"""Audit small within-image probability shifts for the actual soft rewards, separately from argmax decisions."""
import numpy as np
from collections import defaultdict
from compare import OUT,read,dump,boot,calibrate
from multi_view import folds,pred
from orders import advantages
def main():
    l=read(OUT/'inputs/multi_nomination.json');w=l['weights'];m=l['model'];result={};allrows=[]
    for tag in ('fresh','extension'):
        groups=defaultdict(lambda:defaultdict(list))
        for f in folds(tag)[0]:
            new=calibrate(pred(f,m)@np.array([w[2],w[0],w[1]]),f['rk'])
            dest=groups[f['image'],f['seed'],f['pair']]
            for arm,r in [('old',f['old']),('new',new),('R2',f['r2'])]:
                a=advantages(r)
                dest[arm].append((a[:,None]*(f['y']-f['y'].mean(0))).mean(0))
        rows=[dict(tag=tag,image=n,seed=s,pair=p,x={k:np.mean(v,axis=0).tolist() for k,v in arms.items()}) for (n,s,p),arms in groups.items()]
        allrows.extend(rows)
    for tag in ('fresh','extension','all'):
        z=[r for r in allrows if tag=='all' or r['tag']==tag];labs=[r['tag']+'|'+r['pair'] for r in z]
        arrays={k:np.array([r['x'][k] for r in z]) for k in ('new','old','R2')}
        arrays['new_minus_old']=arrays['new']-arrays['old'];arrays['new_minus_R2']=arrays['new']-arrays['R2']
        result[tag]={k:{metric:boot(y[:,i],np.ones(len(z)),labs) for i,metric in enumerate(('joint','mrr','detail'))} for k,y in arrays.items()}
    dump(OUT/'results/reward_credit.json',result);dump(OUT/'records/reward_credit.json',allrows)
if __name__=='__main__':main()
