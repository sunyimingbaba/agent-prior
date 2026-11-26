"""Check both sides of an opposite-fact image pair, and subgroup by development-caption overlap."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from data import ROOT,PREV,OLD,PRIOR_RUN,LIBRARIES,read,load_bank,stream
from new_phrases import load

ARMS=('ordinary','old','old3','switch','local3')


def main():
    known=read(ROOT/'inputs/独立措辞.json')
    fresh=torch.load(ROOT/'inputs/fresh_text.pt',weights_only=True)
    words={s:i for i,s in enumerate(fresh['words'])}
    roster=read(OLD/'inputs/real_roster.json');held={r['rgb_sha256'] for r in read(PRIOR_RUN/'inputs/holdout.json')}
    sources=[torch.load(OLD/'records/text.pt',weights_only=True),torch.load(OLD/'records/covered/bank/text.pt',weights_only=True)]
    labels=set()
    for r in roster:
        if r['rgb_sha256'] in held:continue
        bank=sources[int(r['tag'] in ('saved','extra'))]
        for j in bank['maps'][r['image']]['native']:
            labels.add(tuple(bank['native_tokens'][j].tolist()))
    fold={}; subsets=defaultdict(list);out={}
    for tag in ('v122','v148'):
        native=torch.load(LIBRARIES[tag]/'records/text.pt',weights_only=True)['native_tokens'].tolist()
        p=load(tag);n=load(tag,'neighbor_')
        for row,s,v,bank,vis in stream(tag):
            ix=bank['maps'][row['image']];alts=sorted(set(ix['native'])-{ix['native'][0]})
            if len(alts)<2:continue
            pairs=np.array([ix[k] for k in 'DE'])
            h=vis[:16]@bank['hard'][pairs.ravel()].T
            sides=known[row['attribute']]
            ix2=np.array([[words[a] for a in ss] for ss in sides])
            f=vis[:16]@fresh['features'][ix2.ravel()].double().numpy().T
            f=f.reshape(16,2,2)
            di=np.column_stack((h[:,::2]>h[:,1::2],f[:,:,row['side']]>f[:,:,1-row['side']]))
            for block in (0,1):
                sums=defaultdict(list)
                for held_idx,col in enumerate(alts):
                    k=(row['image'],s,block,held_idx)
                    for a in ARMS:
                        o=(n if a=='local3' else p)[k]['outcomes'][a]
                        sums[a].append(di[8*block+np.array(o['top'])].mean(0))
                        subsets[tag,row['rgb_sha256'],s,tuple(native[col]) in labels,a].append((o['detail'],o['mrr']))
                for a in ARMS:
                    fold[tag,row['pair'],s,block,a,row['side']]=np.mean(sums[a],0)
    for tag_list,name in [(('v122',),'v122'),(('v148',),'v148'),(('v122','v148'),'pooled')]:
        pairs=sorted({(t,p) for t,p,*_ in fold if t in tag_list})
        pairs=[(t,p) for t,p in pairs if (t,p,'a',0,'old',0) in fold and (t,p,'a',0,'old',1) in fold]
        count=len(pairs)
        draws=np.random.default_rng(261126).multinomial(count,np.ones(count)/count,size=20000)
        pairvals={}
        for a in ARMS:
            vals=np.array([[np.mean([fold[t,p,s,b,a,0]*fold[t,p,s,b,a,1] for b in (0,1)],0) for t,p in pairs] for s in 'ab'])
            pairvals[a]=np.stack((vals[:,:,:2].mean(2),vals[:,:,2:].mean(2)),axis=-1)
        stats={}
        for a in ARMS:
            comp=pairvals[a].mean(0)
            delta=comp-pairvals['old'].mean(0)
            stats[a]=dict(level=comp.mean(0).tolist(),vs_old=delta.mean(0).tolist(),
                          intervals=np.quantile((draws@delta)/count,[.025,.975],axis=0).T.tolist())
        strata={}
        for seen in (False,True):
            z={}
            for a in ARMS:
                groups=[vv for (t,_,_,flag,arm),vv in subsets.items() if t in tag_list and seen==flag and arm==a]
                if groups:z[a]=np.mean([np.mean(vv,0) for vv in groups],0).tolist()
            strata[str(seen)]=dict(images=len({rgb for t,rgb,_,flag,_ in subsets if t in tag_list and flag==seen}),levels=z)
        out[name]=dict(pairs=count,measure_order=['DE','FG'],both_sides=stats,held_caption_in_development=strata)
    (ROOT/'results/自然图对和留句核查.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(1);main()
