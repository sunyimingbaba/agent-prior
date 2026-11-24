from fractions import Fraction
import numpy as np

def calibrate(order,ranks):
    # Exact rational arithmetic makes identically ranked proxy slots equal across tie groups.
    values=sorted(Fraction(1,1+int(r)) for r in ranks)
    result=np.empty(len(ranks));offset=0
    for v in np.unique(order):
        mask=order==v;n=int(mask.sum())
        result[mask]=float(sum(values[offset:offset+n],Fraction())/n)
        offset+=n
    return result
def score(sc,kappa):
    rank,native,hard,keep=(np.array([r[k] for r in sc]) for k in ('rank','native','A','keep'))
    r2=calibrate(native,rank)
    floor=float(keep[r2==r2.max()].mean())
    _,dense=np.unique(native+kappa*hard,return_inverse=True)
    r3=calibrate(dense+(len(rank)+1)*(keep+1e-14>=floor),rank)
    return {'R2':r2,'R3':r3},floor
def advantages(total):
    if np.all(total==total[0]):return np.zeros_like(total)
    return (total-total.mean())/(total.std(ddof=1)+1e-8)
