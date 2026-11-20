"""Per-view normalized returns with an optional weakest-view directional score."""
import numpy as np
import torch
from scipy.stats import rankdata

def calculate(margins, condition):
    caption, direction=np.asarray(margins,dtype=float).T
    if len(direction)!=8 or condition not in ('single','separate','joint'):
        raise ValueError('Expected eight trajectories and a registered reward')
    directional=direction.copy()
    if condition=='joint':
        weakest=np.minimum(direction[:4],direction[4:])
        directional=np.tile(weakest,2)
    cuts=[slice(None)] if condition=='single' else [slice(0,4),slice(4,8)]
    scores=[];advantages=[]
    for cut in cuts:
        score=(rankdata(caption[cut])+rankdata(directional[cut]))/2
        reward=torch.tensor(score,dtype=torch.float32)
        advantage=(reward-reward.mean())/(reward.std()+1e-8)
        scores.extend(score.tolist());advantages.extend(advantage.tolist())
    return scores,advantages
