"""Tie-aware retrieval measures from the previous validated UCM protocol."""
import csv, math
from functools import lru_cache
import numpy as np
def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as fp:
        w=csv.DictWriter(fp,list(dict.fromkeys(k for z in rows for k in z)));w.writeheader();w.writerows(rows)
def fixed(x,y):
    a=x.double().numpy();b=y.double().numpy()
    scores=np.zeros((len(a),len(b)),dtype=np.float64)
    for i in range(a.shape[1]):scores+=a[:,i,None]*b[None,:,i]
    return scores
@lru_cache(None)
def hit(t,p,k):return 1-math.comb(t-p,k)/math.comb(t,k)
def recalls(scores,labels):
    result={}
    for k in [1,5,10]:
        boundary=np.partition(scores,-k,axis=1)[:,-k,None]
        above=scores>boundary;same=scores==boundary
        sure=(above&labels).any(1);nt=same.sum(1);cp=(same&labels).sum(1);slots=k-above.sum(1)
        result[k]=np.array([1. if ok else hit(int(n),int(p),int(s)) for ok,n,p,s in zip(sure,nt,cp,slots)])
    return result
def ci(vals):
    x=np.array(vals); sample=np.random.default_rng(20261120).integers(len(x),size=(20000,len(x)))
    a,b=np.quantile(x[sample].mean(1),[.025,.975])
    return dict(mean=float(x.mean()),lo=float(a),hi=float(b),n=len(x))
