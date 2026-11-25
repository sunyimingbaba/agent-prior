"""Exact expected bidirectional recalls when one tied winner is drawn independently per source image."""
from functools import lru_cache
from itertools import product
import numpy as np
from single_scores import recalls

@lru_cache(maxsize=None)
def quadrature(tied_groups):
    nodes,weights=np.polynomial.legendre.leggauss((tied_groups+2)//2)
    return (nodes+1)/2,weights/2

def cdf_at_ks(prob):
    cutoff=np.array([1,5,10])-int((prob>=1-1e-14).sum())
    uncertain=prob[(prob>1e-14)&(prob<1-1e-14)]
    mass=np.zeros(10);mass[0]=1
    for x in uncertain:
        old=mass.copy();mass*=1-x;mass[1:]+=x*old[:-1]
    cum=np.r_[0.,mass.cumsum()]
    return cum[np.clip(cutoff,0,10)]

def expected(scores,weights,owners,labels):
    n=len(weights);sizes=[len(w) for w in weights]
    groups=np.repeat(np.arange(n),sizes);flat=np.concatenate(weights)
    assert np.allclose(np.bincount(groups,weights=flat),1,rtol=0,atol=1e-12)
    relevance=groups[:,None]==owners[None,:]
    i2t=np.stack(list(recalls(scores,relevance).values()),axis=1)
    by_image=np.stack([np.bincount(groups,weights=i2t[:,i]*flat,minlength=n) for i in range(3)],axis=1)
    t2i=np.zeros((len(owners),3))
    for j,correct in enumerate(owners):
        s=scores[:,j];pos=groups==correct
        vals,inv=np.unique(s[pos],return_inverse=True)
        for target,share in zip(vals,np.bincount(inv,weights=flat[pos])):
            greater=np.bincount(groups,weights=flat*(s>target),minlength=n);greater[correct]=0.
            equal=np.bincount(groups,weights=flat*(s==target),minlength=n);equal[correct]=0.
            n_ties=int((equal>0).sum())
            if n_ties==0:
                t2i[j]+=share*cdf_at_ks(greater)
            else:
                nodes,w=quadrature(n_ties)
                t2i[j]+=share*sum(ww*cdf_at_ks(greater+u*equal) for u,ww in zip(nodes,w))
    reverse=np.stack([t2i[owners==i].mean(0) for i in range(n)])
    hits=np.hstack((by_image,reverse))
    assert np.all(hits>=-1e-12) and np.all(hits<=1+1e-12)
    return hits

def self_check():
    rng=np.random.default_rng(2741);n=12;owners=np.repeat(np.arange(n),2)
    weights=[np.array([.25,.75]) if i<3 else np.ones(1) for i in range(n)]
    scores=rng.integers(0,5,(sum(map(len,weights)),len(owners))).astype(float)
    mask=np.arange(n)[:,None]==owners[None,:]
    obs=expected(scores,weights,owners,np.arange(n))
    start=np.cumsum([0]+[len(w) for w in weights[:-1]]);total=np.zeros_like(obs)
    for slots in product(*(range(len(w)) for w in weights)):
        mat=scores[start+np.array(slots)]
        a=np.stack(list(recalls(mat,mask).values()),1)
        b=np.stack(list(recalls(mat.T,mask.T).values()),1).reshape(n,2,3).mean(1)
        total+=np.prod([w[j] for w,j in zip(weights,slots)])*np.hstack((a,b))
    error=float(np.max(abs(total-obs)));assert error<1e-12
    return error
