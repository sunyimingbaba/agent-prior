"""Score paired image-prior swaps using the same held-out questions and gallery as the anchor."""
import sys
import numpy as np
from candidate_guard import ROOT,HIST,load,read,dump,boot
from multi_caption import digest
from recheck_reward import rr

def run(tag):
    independent=tag=='independent'
    base=ROOT/('reward_confirm' if independent else 'guard')
    gs=read(base/'inputs/panel.json')
    bank=load(base/'records/text.pt')
    native=bank['native'].double().numpy();hard=bank['hard'].double().numpy()
    hashes=[digest(v) for v in bank['native_tokens'].tolist()]
    odd=np.array([int(h,16)%2==1 for h in hashes])
    scores={};same={}
    for seed in 'ab':
        anchor=load(ROOT/'pilot/vectors'/(seed+'_start_reward_anchor.pt') if independent else HIST/'pilot/vectors'/(seed+'_start_fresh.pt'))
        swap=load(ROOT/'records'/('swapped_'+tag+'_'+seed+'.pt'))
        assert swap['images']==anchor['images']==[c['image'] for c in gs] and swap['adapter']==anchor['adapter']
        pairs={c['pair']:{r['image'] for r in gs if r['pair']==c['pair']} for c in gs}
        proofs=dict(zip(anchor['images'],anchor['sources']))
        assert swap['sources']==[proofs[next(iter(pairs[r['pair']]-{r['image']}))] for r in gs]
        same[seed]=swap['matched_tool_returns']
        for name,z in [('own',anchor),('swapped',swap)]:
            vis=z['features'].double().numpy();sim=vis@native.T;hs=vis@hard.T;values=[]
            for i,c in enumerate(gs):
                ix=bank['maps'][c['image']];p=ix['native'];alt=sorted(set(p)-{p[0]},key=hashes.__getitem__)
                held=alt[1:] if len(alt)>1 else alt;neg=odd.copy();neg[p]=False
                values.append([np.mean([hs[i,ix[k][0]]>hs[i,ix[k][1]] for k in 'CDE']),rr(np.array([sim[i,held].max()]),sim[i,neg][None]).item()])
            scores[(seed,name)]=np.array(values)
    z=np.array([scores[(seed,'own')]-scores[(seed,'swapped')] for seed in 'ab'])
    also=np.array(same['a'])&np.array(same['b'])
    out={}
    for name,mask in [('all',np.ones(len(gs),bool)),('both_seeds_same_tool_returns',also)]:
        labels=[r['pair'] for r,m in zip(gs,mask) if m]
        out[name]=dict(images=int(mask.sum()),seeds_same_tools={s:sum(same[s]) for s in 'ab'},levels={m:np.mean([scores[(s,m)][mask] for s in 'ab'],axis=(0,1)).tolist() for m in ('own','swapped')},difference={k:dict(**boot(z[:,mask,j].mean(0),labels),seeds=z[:,mask,j].mean(1).tolist()) for j,k in enumerate(('detail','rr'))}) if mask.any() else dict(images=0)
    dump(ROOT/'results'/('swap_'+tag+'.json'),out)
if __name__=='__main__':run(sys.argv[1])
