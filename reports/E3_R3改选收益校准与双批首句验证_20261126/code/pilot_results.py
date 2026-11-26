"""Single-pass evaluation on an identical frozen reader, with paired natural-unit intervals."""
from pathlib import Path
import json
import numpy as np
import torch
from data import ROOT, read, rr, sha
from single_scores import RFPO,FORM,calculate,summary_by_seed,fixed
from exact_scores import paired

def get(tag,arm,s,rows):
    if arm=='retained':base=ROOT.parent/'E3_双输入同预算RFPO_20261126/reference/evaluate'/(s+'_retained')
    else:base=ROOT/'evaluate'/(s+'_'+arm)
    target=base/(tag+'.pt')
    assert sha(target)==read(base/tag/'complete.json')['features']
    z=torch.load(target,map_location='cpu',weights_only=True)
    assert [r['image'] for r in rows]==z['images']
    if arm!='retained':assert z['adapter']==read(ROOT/'models'/(s+'_'+arm)/'complete.json')['final']
    return z['features']


def main():
    arms=read(ROOT/'inputs/pilot/decision.json')['train_arms']+['retained']
    panels=read(RFPO/'inputs/panel.json');gallery=read(FORM/'inputs/ucm.json')
    hard=torch.load(RFPO/'records/text.pt',map_location='cpu',weights_only=True)
    native=torch.load(FORM/'results/ucm_text.pt',map_location='cpu',weights_only=True)
    cols=native['caption_order'].numpy().reshape(210,5)
    index={r['rgb_sha256']:i for i,r in enumerate(gallery)}
    overlap=[r for r in panels if r['rgb_sha256'] in index];assert len(overlap)==106
    tags={'official':gallery,'primary':panels,'first106':overlap}
    data={t:{a:{} for a in arms} for t in tags}
    for arm in arms:
        for s in 'ab':
            for tag,rows in [('primary',panels),('official',gallery)]:
                vs=get(tag,arm,s,rows)
                val=calculate(tag,vs,rows)
                if tag=='primary':val['DE']=(val['D']+val['E'])/2
                data[tag][arm][s]=val
                if tag!='official':continue
                sd=fixed(vs,hard['hard']);ns=fixed(vs,native['native']);hs=[];rs=[]
                for r in overlap:
                    j=index[r['rgb_sha256']];ix=hard['maps']['primary'][r['image']]
                    hs.append(np.mean([sd[j,ix[k][0]]>sd[j,ix[k][1]] for k in 'DE']))
                    pos=cols[j];alts=sorted(set(pos)-{pos[0]});assert alts
                    mask=np.ones(ns.shape[1],dtype=bool);mask[pos]=False
                    rs.append(np.mean([float(rr(ns[j,k],ns[j,mask])) for k in alts]))
                data['first106'][arm][s]=dict(DE=np.array(hs),mrr=np.array(rs))
    out={}
    for tag,rows in tags.items():
        labels=[r.get('pair',r['image']) if tag!='official' else r['image'] for r in rows]
        out[tag]=dict(images=len(rows),levels={a:summary_by_seed(data[tag][a],labels) for a in arms},
                      delta={b:{a:paired(data[tag][a],data[tag][b],labels) for a in arms if a!=b} for b in arms})
    (ROOT/'results/单次推理.json').write_text(json.dumps(out,indent=2)+'\n')
    for s in 'ab':
        records={a:read(ROOT/'models'/(s+'_'+a)/'steps.json') for a in arms if a!='retained'}
        assert all(len(v)==24 for v in records.values())
        base=records['ordinary'][0]
        assert all(v[0]['states']==base['states'] and v[0]['paths']==base['paths'] and v[0]['winners']==base['winners'] for v in records.values())


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
