import numpy as np
EPS=1e-10
TINY=1e-10

def best(keys, eligible):
    if len(eligible) == 0:
        return []
    top = max(keys[i] for i in eligible)
    return [int(i) for i in eligible if keys[i] == top]


def centered(v,ref):
    return v-v[ref].mean(0)


def gains_features(h,reference,similar,rrs,gaps,first,near,spread,base):
    # Per-phrase order is intentionally symmetric in the new model.
    cross=centered(h,reference)
    positive=h>0
    pair=np.column_stack((h.min(1),h.max(1),h.mean(1),positive.mean(1),
                          cross.min(1),cross.max(1),cross.mean(1),(cross>0).mean(1),
                          (centered(positive.astype(float),reference)).mean(1)))
    def stats(values):
        delta=centered(values,reference)
        return np.column_stack((values.mean(1),values.min(1),delta.min(1),delta.max(1),
                                delta.mean(1),(delta>1e-12).mean(1),(delta< -1e-12).mean(1)))
    common=np.column_stack((stats(rrs),stats(gaps),stats(similar),stats(near),first,
                            np.full(8,spread),base))
    combined=np.column_stack((pair,common))
    relative=centered(combined,reference)
    # Give the linear model access to baseline difficulty near either decision boundary.
    baseline_margin=abs(h[reference].mean())
    certainty=np.exp(-baseline_margin/.025)
    room=1-rrs[reference].mean()
    extras=np.column_stack((cross.mean(1)*certainty,cross.min(1)*certainty,
                            centered(gaps.mean(1),reference)*room,centered(near.mean(1),reference)*room))
    return np.column_stack((combined,relative,extras))


def old_choices(pred,p):
    gains=pred-pred[p['ref']].mean(0)
    eligible=np.flatnonzero((gains>1e-12).all(1))
    return best(list(zip(gains.min(1),p['exact'])),eligible) or p['ref']

def choose(p,pred,old,weight,mode):
    gains=pred-pred[p['ref']].mean(0)
    fine=(gains[:,0]+gains[:,1])/2
    rank=gains[:,2]
    utility=fine+weight*rank+.5*gains[:,3]
    allowed=utility>EPS
    if mode=='positive':
        allowed &= (fine>EPS)&(rank>EPS)
    elif mode=='veto':
        return old if utility[old].mean()>EPS else p['ref']
    eligible=np.flatnonzero(allowed)
    return best(list(zip(np.round(utility,10),p['exact'])),eligible) or p['ref']

def unit(v):
    return v/max(np.linalg.norm(v),1e-12)

def quantiles_of_scores(scores,neg):
    sorted_neg=np.sort(neg,axis=1)
    return np.vstack([1/(1+len(pool)-np.searchsorted(pool,x,side='right')) for x,pool in zip(scores,sorted_neg)])

def stats_matrix(score,neg):
    pseudo=quantiles_of_scores(score,neg)
    maxneg=neg.max(1)
    eighth=np.partition(neg,-8,axis=1)[:,-8:].mean(1)
    return np.column_stack((score.mean(1),score.max(1),score.min(1),pseudo.mean(1),pseudo.max(1),
                            pseudo.min(1),score.mean(1)-maxneg,score.mean(1)-eighth))

def decide(p,old,rank,weight,ordering):
    diff=old[:,0]-old[p['ref'],0].mean()
    rank=rank-rank[p['ref']].mean()
    score=diff+weight*rank
    valid=score>TINY
    if ordering=='both':valid&=(diff>TINY)&(rank>TINY)
    return best(list(zip(np.round(score,10),p['exact'])),np.flatnonzero(valid)) or p['ref']
