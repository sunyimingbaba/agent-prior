"""Attribute-directed hidden-state scoring and caption-protected redistribution of R2."""
from fractions import Fraction
import numpy as np


def unit(values):
    x=np.asarray(values,dtype=np.float64)
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.all((norms>0)&np.isfinite(norms)):
        raise ValueError("invalid hidden-state norm")
    return x/norms


def calibrated_scores(hidden,side,direction):
    if side not in (0,1):
        raise ValueError("invalid side")
    return (1-2*side)*np.sum(unit(hidden)*np.asarray(direction,dtype=np.float64),axis=-1)


def redistribute(r2,score,other_rr,visible=None):
    base=np.asarray(r2,dtype=np.float64)
    signal=np.asarray(score,dtype=np.float64)
    rr=np.asarray(other_rr,dtype=np.float64)
    if not np.all(np.isfinite(signal)) or rr.shape[0]!=len(base) or base.shape!=signal.shape:
        raise ValueError("bad score group")
    cols=list(range(rr.shape[1]) if visible is None else visible)
    result=base.copy()
    if not cols:
        return result
    top=np.flatnonzero(base==base.max())
    thresholds=[sum(Fraction.from_float(float(v)) for v in rr[top,k]) for k in cols]
    challengers=[i for i in range(len(base)) if base[i]<base[top[0]] and signal[i]>max(signal[top]) and
                 all(Fraction.from_float(float(rr[i,k]))*len(top)>=t for k,t in zip(cols,thresholds))]
    if not challengers:
        return result
    selected=np.array([*top,*challengers])
    rank=selected[np.argsort(signal[selected],kind="stable")]
    ordered=np.sort(base[selected])
    for v in np.unique(signal[rank]):
        same=signal[rank]==v
        result[rank[same]]=ordered[same].mean()
    return result
