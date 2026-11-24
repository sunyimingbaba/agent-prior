"""Secondary comparison with the actual single greedy R2 output on exactly the same gallery."""
import sys
import numpy as np
from candidate_guard import ROOT,HIST,load,read,dump,boot
from recheck_reward import rr
from multi_caption import digest

def run(tag):
    assert tag in ('fresh','independent')
    base=ROOT/('reward_confirm' if tag=='independent' else 'guard')
    panel=read(base/'inputs/panel.json')
    bank=load(base/'records/text.pt')
    nat=bank['native'].double().numpy(); hard=bank['hard'].double().numpy()
    hashes=[digest(row) for row in bank['native_tokens'].tolist()]
    odd=np.array([int(h[-1],16)%2==1 for h in hashes])
    cached={(r['image'],r['seed']):r['scores'] for r in read(ROOT/'results'/('multi_'+tag+'_rows.json'))}
    stat={s:[] for s in 'ab'};comparisons={}
    for seed in 'ab':
        file=(ROOT/'pilot/vectors'/(seed+'_start_reward_anchor.pt') if tag=='independent' else HIST/'pilot/vectors'/(seed+'_start_fresh.pt'))
        pack=load(file)
        assert pack['images']==[c['image'] for c in panel]
        vis=pack['features'].double().numpy()
        for c,v in zip(panel,vis):
            ix=bank['maps'][c['image']];s=v@nat.T;h=v@hard.T
            pos=ix['native'];alt=sorted(set(pos)-{pos[0]},key=hashes.__getitem__);held=alt[1:] if len(alt)>1 else alt
            neg=odd.copy();neg[pos]=False
            detail=np.mean([h[ix[k][0]]>h[ix[k][1]] for k in 'CDE'])
            ordinary=rr(np.array([max(s[held])]),s[neg][None]).item()
            stat[seed].append((detail,ordinary))
    for name in ('R2','R3','guard'):
        ds=[]
        for seed in 'ab':
            a=[cached[(c['image'],seed)][name] for c in panel]
            ds.append(np.array([[np.mean([r[k] for k in 'CDE']),r['rr']] for r in a])-np.array(stat[seed]))
        ds=np.array(ds)
        comparisons[name]={k:dict(**boot(ds[:,:,j].mean(0),[c['pair'] for c in panel]),seeds=ds[:,:,j].mean(1).tolist()) for j,k in enumerate(('detail','rr'))}
    dump(ROOT/'results'/('single_vs_candidates.json' if tag=='independent' else 'single_vs_candidates_fresh.json'),dict(single_levels=np.mean(list(stat.values()),axis=(0,1)).tolist(),compared_to_single=comparisons))
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'independent')
