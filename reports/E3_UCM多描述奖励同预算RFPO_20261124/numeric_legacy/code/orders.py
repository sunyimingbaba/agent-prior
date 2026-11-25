import numpy as np
def calibrate(order,proxy):
    values=np.sort(proxy);scores=np.empty(len(proxy));offset=0
    for v in np.unique(order):
        mask=order==v;n=int(mask.sum());scores[mask]=values[offset:offset+n].mean();offset+=n
    return scores
def score(sc,kappa):
    rank,native,hard,keep=(np.array([r[k] for r in sc]) for k in ('rank','native','A','keep'))
    proxy=1/(1+rank)
    r2=calibrate(native,proxy)
    floor=float(keep[r2==r2.max()].mean())
    _,dense=np.unique(native+kappa*hard,return_inverse=True)
    r3=calibrate(dense+(len(rank)+1)*(keep+1e-14>=floor),proxy)
    return {'R2':r2,'R3':r3},floor
