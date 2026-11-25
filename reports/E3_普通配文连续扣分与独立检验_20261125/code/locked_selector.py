"""Numeric training-side scorer. Shared feature functions are copied verbatim from the frozen multi-view model."""
import numpy as np
from collections import defaultdict

def neighbors(ctx,visible,pos,t):
    center=t[visible].mean(0);proto,ds=ctx
    top=np.argsort(-(proto@center),kind='stable')[:12];exclude=set(pos);freq=defaultdict(float)
    for i in top:
        use=[j for j in ds[i][1] if j not in exclude]
        for j in use:freq[j]+=1/len(use)
    if not freq:return [],[]
    ix=list(freq);w=np.array([freq[j] for j in ix]);return ix,w/w.sum()

def candidate_features(sim,ranks,neg,h,v,neigh_sim,neigh_rank):
    marg=sim-neg[:,None];hard=h.mean(1);rrmean=ranks.mean(1)
    basic=np.column_stack([sim,marg,ranks,marg.mean(1),marg.min(1),ranks.min(1),rrmean,
                           neigh_sim,neigh_sim-neg,neigh_rank,h,hard,h.min(1),h.max(1),
                           (h>0).astype(float),np.mean(h>0,axis=1),np.tanh(50*h),np.tanh(50*hard)])
    change=basic-basic[-1];dist=np.linalg.norm(v-v[-1],axis=1)
    hard_gain=hard-hard[-1];native_gain=rrmean-rrmean[-1];ng=neigh_rank-neigh_rank[-1]
    inter=np.column_stack([dist,native_gain*hard_gain,ng*hard_gain,(native_gain>0)&(hard_gain>0),
                           (ng>0)&(hard_gain>0),np.minimum(native_gain,np.tanh(50*hard_gain)),
                           (native_gain>0)&(hard>0),np.minimum(ng,np.tanh(50*hard_gain))])
    return np.column_stack([basic,change,inter])[:16]

def ordinal_prediction(model,w,x):
    outcome=(x/np.asarray(model['scale']))@np.asarray(model['coef']).T
    return outcome@np.array([w[2],w[0],w[1]])
