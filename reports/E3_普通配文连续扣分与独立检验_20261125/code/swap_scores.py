"""Separate the image-conditioned part of the fitted residual from a shared readout correction."""
import numpy as np
from head_scores import OUT,HEAD,LOCK,FORM,RFPO,read,load,dump,detailed,counts,summary,compare,sha

def main():
    allout={}
    for tag in ('fresh','extension','primary','official'):
        rows=read((FORM/'inputs/ucm.json' if tag=='official' else (RFPO/'inputs/panel.json' if tag=='primary' else OUT/'inputs'/(tag+'.json'))))
        b=load(FORM/'results/ucm_text.pt' if tag=='official' else (RFPO if tag=='primary' else OUT)/'records/text.pt')
        levels={};arms=['baseline','hard','swapped','constant']
        for arm in arms:
            levels[arm]={}
            for s in 'ab':
                z=load(HEAD/'test'/tag/s/(arm+'.pt'));assert z['nomination']==sha(LOCK) and z['images']==[c['image'] for c in rows]
                if arm in ('swapped','constant'):
                    assert all(i!=k for i,k in enumerate(z['mapping']))
                    assert all((rows[i]['pair']==rows[j]['pair'] if tag!='official' else (int(rows[i]['image'][:-4])-1)//100==(int(rows[j]['image'][:-4])-1)//100) for i,j in enumerate(z['mapping']))
                if tag=='official':
                    _,x,ds=counts(z['ordinary'],b,'native');levels[arm][s]=x
                else:levels[arm][s]=detailed(z['scene'],b,rows,b['maps']['primary'] if tag=='primary' else b['maps']).mean(1)
        links=[('hard','swapped'),('hard','constant'),('swapped','baseline'),('constant','baseline')]
        if tag=='official':
            vals={arm:summary(np.mean(list(z.values()),0),ds)['mR']['mean'] for arm,z in levels.items()}
            change={a+'-'+b:summary(np.stack([levels[a][s]-levels[b][s] for s in 'ab']).mean(0),ds)['mR'] for a,b in links}
        else:
            vals={arm:float(np.mean(list(z.values()))) for arm,z in levels.items()};labs=[r['pair'] for r in rows]
            change={a+'-'+b:compare(levels[a],levels[b],labs) for a,b in links}
        allout[tag]=dict(levels=vals,contrasts=change)
    dump(OUT/'results/head_control.json',allout)
if __name__=='__main__':main()
