"""Check first-round stochastic actions against the existing sampler and the greedy prior against deployment."""
import sys
from pathlib import Path
import torch
from run import ROOT,PREV
from runtime import read,dump,table,rgb,pool,setup
from policy import policy
from eval_io import weights
from train import sample
from rollout import rollout_one,SYSTEM_PROMPT
@torch.inference_mode()
def main(seed):
    setup();model,proc=policy(weights(seed+'_deployed')[0])
    info=read(ROOT/'inputs/lock.json')['rows'][0];nm=info['image']
    c=pool()[nm];im=rgb(table(),c);f=ROOT/'records'/seed;gs=read(f/(nm+'.json'))['groups']
    h=torch.load(f/(nm+'.pt'),weights_only=True);out={}
    for mode,g in gs.items():
        torch.manual_seed(10000000 if seed=='a' else 20000000)
        _,draws=sample(model,proc,im,g['query'],None)
        assert [d[0] for d in draws]==[path[0] for path in g['actions'][:8]]
        prior,n,_=rollout_one(model,proc,im,g['query'],{'scene_classify':lambda _:c['feedback']},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
        assert prior.cpu().equal(h[mode][-1]) and n==len(g['greedy_tools'])
        out[mode]=dict(eight_first_rounds_exact=True,greedy_prior_exact=True)
    dump(ROOT/'results'/('replay_'+seed+'.json'),out)
if __name__=='__main__':main(sys.argv[1])
