"""Keep the target pixels and text fixed and replace only the original R2 image prior."""
import sys
from collections import defaultdict
import torch
from env_pair import ROOT,HIST,setup,read,table,rgb,savept,sha,weight
from readouts import load_model

@torch.inference_mode()
def run(tag):
    setup()
    torch.set_num_threads(3)
    independent = tag=='independent'
    directory = ROOT/'reward_confirm' if independent else HIST
    rows = read(directory/'inputs/panel.json')
    pair = defaultdict(list)
    for c in rows:pair[c['pair']].append(c)
    opposite = {}
    for left,right in pair.values():
        assert left['scene_query']==right['scene_query'] and left['side']!=right['side']
        opposite[left['image']]=right;opposite[right['image']]=left
    t=table()
    m=load_model(False)
    for seed in 'ab':
        start=seed+'_start'
        parent=ROOT if independent else HIST
        split='reward_anchor' if independent else 'fresh'
        folder=parent/'pilot/greedy'/start/split
        proof=read(folder/'complete.json')
        assert proof['adapter']==weight(seed)
        priors={}
        for c in rows:
            nm=c['image'];f=folder/(nm+'.pt')
            assert sha(f)==proof['files'][nm]
            r=torch.load(f,weights_only=True)
            assert r['query']==c['scene_query'] and r['rgb']==c['rgb_sha256'] and r['adapter']==proof['adapter']
            priors[nm]=r
        feats=[];same=[]
        for c in rows:
            nm=c['image'];other=opposite[nm]['image']
            s=priors[other];same.append(priors[nm]['tools']==s['tools'])
            v=m.img_embeds(m.preprocess(rgb(t,c))[None],agent_hidden=s['prior'].float()[None])['image_feats'][0]
            assert abs(v.norm().item()-1)<1e-4
            feats.append(v)
        savept(ROOT/'records'/('swapped_'+tag+'_'+seed+'.pt'),dict(images=[c['image'] for c in rows],features=torch.stack(feats),matched_tool_returns=same,adapter=proof['adapter'],sources=[proof['files'][opposite[c['image']]['image']] for c in rows]))
if __name__=='__main__':run(sys.argv[1])
