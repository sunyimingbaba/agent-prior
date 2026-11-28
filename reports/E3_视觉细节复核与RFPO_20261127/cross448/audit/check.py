"""Reconstruct image and caption recall using exact probability of a relevant tied item."""
from pathlib import Path
import json
from math import comb
import hashlib
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parents[1];FORM=REP/'E3-2_UCM与RSITMD独立复验_20261124'
def read(p):return json.loads(p.read_bytes())
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def hits(s,positive):
    best=max(s[positive]);above=int(np.count_nonzero(s>best));equal=int(np.count_nonzero(s==best))
    npos=int(np.count_nonzero(s[positive]==best))
    def value(k):
        ranks=k-above
        if ranks<=0:return 0.
        if ranks>equal-npos:return 1.
        return 1-comb(equal-npos,ranks)/comb(equal,ranks)
    return np.array([value(k) for k in (1,5,10)])

def by_image(v,t,mode):
    ids=t['caption_order'].reshape(-1,5).numpy()
    sim=v.double().numpy()@t['native'].double().numpy().T
    n=len(ids)
    keep=[(i,k,int(cid)) for i,caps in enumerate(ids) for k,cid in enumerate(caps) if mode=='native' or cid!=caps[0]]
    col=np.array([key for _,_,key in keep]);owner=np.array([i for i,_,_ in keep]);use=np.unique(owner)
    # The caption-to-image task keeps every original gallery image as a possible distractor.
    itr=[];ttr=[]
    for i in use:
        row=sim[i,col].copy();true=np.flatnonzero(owner==i)
        if mode!='native':row[col==ids[i,0]]=-np.inf
        itr.append(hits(row,true))
        ttr.append(np.sum([hits(sim[:,col[j]],[i]) for j in true],axis=0))
    return np.column_stack((itr,ttr)),np.bincount(owner,minlength=n)[use],use

def pooled(total,owners,texts):
    rate=total/np.stack([owners,owners,owners,texts,texts,texts],axis=-1)
    return np.concatenate([rate,rate[...,:3].mean(-1,keepdims=True),rate[...,3:].mean(-1,keepdims=True),rate.mean(-1,keepdims=True)],axis=-1)

def baseline():
    gallery=read(FORM/'inputs/rsitmd.json');t=load(FORM/'results/rsitmd_text.pt')
    assert len(gallery)==len(set(r['rgb_sha256'] for r in gallery))==448 and t['caption_order'].shape==(2240,)
    orig=read(FORM/'results/paired_results.json')['rsitmd']
    z=load(FORM/'results/rsitmd_a_R2.pt')
    assert z['manifest']==sha(FORM/'inputs/rsitmd.json')
    out={}
    for mode in ('native','alternatives'):
        arr,c,im=by_image(z['features'],t,mode);hits=100*pooled(arr.sum(0),np.array(len(im)),np.array(c.sum()))
        gold=orig['all_'+mode]['levels']['a_R2']
        for x,k in zip(hits[:6],('I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10')):
            assert abs(gold[k]-x)<1e-10
        assert abs(gold['mR']-hits[-1])<1e-10
        out[mode]={'images':len(im),'captions':int(c.sum()),'distractor_images':448,'reference_mR_percent':hits[-1],
                  'excluded_source_images':[r['image'] for i,r in enumerate(gallery) if i not in im]}
    assert out['alternatives']['images']==446 and out['alternatives']['captions']==1739
    (ROOT/'audit/历史图库与六项召回核对.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(1)
    baseline()
