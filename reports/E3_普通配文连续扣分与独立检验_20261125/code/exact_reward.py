import numpy as np
from fractions import Fraction
def payouts(order):
    values=[Fraction(1,k) for k in range(len(order),0,-1)]
    groups={}
    for i,x in enumerate(order):groups.setdefault(x,[]).append(i)
    r=np.zeros(len(order));offset=0
    for key in sorted(groups):
        ids=groups[key];r[ids]=float(sum(values[offset:offset+len(ids)])/len(ids));offset+=len(ids)
    return r
def exact_rewards(sc):
    rr=[[Fraction(x[0],x[1]) for x in r] for r in sc['rr']]
    detail=np.asarray(sc['detail']);anchor=rr[-1]
    mrr=[sum(v)/len(v) for v in rr[:-1]]
    joint=[Fraction(sum(a>b for a,b in zip(v,anchor)),len(v)) if d>detail[-1] else Fraction() for v,d in zip(rr[:-1],detail[:-1])]
    return {'ordinary':payouts(mrr),'joint':payouts(list(zip(joint,mrr)))},[float(x) for x in joint]
