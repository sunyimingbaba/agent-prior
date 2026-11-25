"""Inputs for all selectors come only from reward-visible captions and hard comparison A."""
import numpy as np

def features(sim, rr, neg_max, hard, visual):
    margin = sim - neg_max[:, None]
    raw = np.column_stack([sim, margin, rr, margin.mean(1), margin.min(1),
                           rr.mean(1), rr.min(1), hard])
    delta = raw - raw[-1]
    better = (delta[:-1, -1] > 0) & (rr[:-1].mean(1) > rr[-1].mean())
    return np.column_stack([raw[:-1], delta[:-1], np.linalg.norm(visual[:-1]-visual[-1],axis=1), better])

def model_order(x, model):
    z = (x-np.asarray(model['mean']))/np.asarray(model['scale'])
    return z@np.asarray(model['coef'])+model['intercept']
