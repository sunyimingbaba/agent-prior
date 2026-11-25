"""Reconstruct held-out and reward-side readouts for the first locked pair across both weights."""
import sys
from pathlib import Path
import hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'multicap_rfpo_20261124'
sys.path.insert(0,str(PREV/'code'))
from runtime import REBUILD,OLD,pool,table,rgb,read,dump,setup
from readouts import load_model,scorer_for
from replay_read import reciprocal

@torch.inference_mode()
def main():
    setup();cup=load_model(False);proxy=scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    cases=pool();t=table();n=0;err=0.
    for info in read(ROOT/'inputs/lock.json')['rows'][:2]:
        nm=info['image'];c=cases[nm];bank=torch.load(OLD/c['sub']/'records/text.pt',weights_only=True)
        ix=bank['maps'][nm];pos=ix['native']
        hashes=[hashlib.sha256(','.join(str(x) for x in ids).encode()).hexdigest() for ids in bank['native_tokens'].tolist()]
        alt=sorted(set(pos)-{pos[0]},key=hashes.__getitem__)
        train=[pos[0]]+(alt[:1] if len(alt)>=2 else [])
        held=alt[1:] if len(alt)>=2 else alt
        neg1=[i for i,k in enumerate(hashes) if int(k,16)%2==0 and i not in pos];neg2=[i for i,k in enumerate(hashes) if int(k,16)%2 and i not in pos]
        pix=cup.preprocess(rgb(t,c))[None]
        for seed in 'ab':
            path=ROOT/'records'/seed
            states=torch.load(path/(nm+'.pt'),weights_only=True)
            groups=read(path/(nm+'.json'))['groups']
            for mode,hs in states.items():
                cache={}
                for h,expect in zip(hs,groups[mode]['scores'],strict=True):
                    key=h.float().numpy().tobytes()
                    if key not in cache:
                        v=cup.img_embeds(pix,agent_hidden=h.float()[None])['image_feats'][0].double()
                        ss=(bank['native'].double()@v).numpy();hard=(bank['hard'].double()@v).numpy()
                        d=dict(rank=proxy.score(h.float().tolist(),c['row'])[0],native=ss[pos[0]]-max(ss[neg1]),
                            keep=np.mean([reciprocal(ss[j],ss[neg1]) for j in train]),rr=reciprocal(max(ss[held]),ss[neg2]))
                        d.update({k:hard[ix[k][0]]-hard[ix[k][1]] for k in 'ABCD' if k in ix});cache[key]=d
                    got=cache[key];assert got['rank']==expect['rank']
                    err=max(err,max(abs(val-expect[k]) for k,val in got.items()));n+=1
    assert n==136 and err<1e-9
    dump(ROOT/'results/score_recheck.json',dict(states=n,proxy_ranks_exact=True,max_absolute_error=err))
if __name__=='__main__':main()
