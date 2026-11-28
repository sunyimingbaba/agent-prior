"""Compare the four frozen DPO outputs using the historical and independent retrieval implementations."""
from collections import Counter
from math import comb
from pathlib import Path
import gzip
import hashlib
import json
import sys
import tarfile

import numpy as np
import torch

from prepare import ROOT,FORM,X,read,dump,sha
sys.path.insert(0,str(X/"code"))
from transfer_scores import counts,scalar_values,summary

KEYS=("I2T_R1","I2T_R5","I2T_R10","T2I_R1","T2I_R5","T2I_R10","I2T","T2I","mR")
MODES=("alternatives","native")

def tie_hits(scores, positives):
    p=scores[positives];best=p.max()
    above=int((scores>best).sum()); equal=int((scores==best).sum()); good=int((p==best).sum())
    out=[]
    for k in (1,5,10):
        left=k-above
        out.append(0. if left<=0 else 1. if left>equal-good else 1.-comb(equal-good,left)/comb(equal,left))
    return out

def independent(v,bank,mode):
    order=bank["caption_order"].numpy().reshape(448,5)
    scores=v.double().numpy()@bank["native"].double().numpy().T
    ix,slot=np.where(order!=order[:,:1]) if mode=="alternatives" else np.indices(order.shape).reshape(2,-1)
    text=order[ix,slot]
    mat=scores[:,text]
    valid,den=np.unique(ix,return_counts=True)
    forward=[]
    for i in valid:
        x=mat[i].copy()
        if mode=="alternatives":
            x[text==order[i,0]]=-np.inf
        forward.append(tie_hits(x,ix==i))
    backward=np.zeros((448,3))
    for j,i in enumerate(ix):
        backward[i]+=tie_hits(mat[:,j],[i])
    return valid,np.column_stack((forward,backward[valid])),den

def independent_change(arr,den,expected):
    n=len(den);rng=np.random.default_rng(20261125);boot=[]
    for block in range(40):
        draw=rng.integers(n,size=(500,n))
        weights=np.stack([np.bincount(row,minlength=n) for row in draw])
        totals=weights@arr
        directions=np.column_stack(((totals[:,:3]/n).mean(1),(totals[:,3:]/(weights@den)[:,None]).mean(1)))
        boot.append(np.column_stack((directions,directions.mean(1))))
    ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0).T
    for i,k in enumerate(("I2T","T2I","mR")):
        np.testing.assert_allclose(ci[i],expected[k]["interval"],rtol=0,atol=1e-12)

def archive(tag,folder,files):
    path=ROOT/"archives"/(tag+".tar.gz")
    if not path.exists():
        tmp=path.with_suffix(".part")
        with tarfile.open(tmp,"w:gz",compresslevel=4) as tf:
            for n in sorted(files):tf.add(folder/n,arcname=n)
        tmp.replace(path)
    found={}
    with tarfile.open(path,"r|gz") as tf:
        for member in tf:
            assert member.isfile() and member.name not in found
            found[member.name]=hashlib.file_digest(tf.extractfile(member),"sha256").hexdigest()
    assert found==files
    return dict(sha256=sha(path),files=len(files),bytes=path.stat().st_size)

def inspect(tag,rows,tools,lock):
    folder=ROOT/"records"/tag; cp=read(folder/"complete.json")
    sig=lock["policies"][tag]["adapter"]
    assert cp["adapter"]==sig and cp["lock"]==sha(ROOT/"inputs/server_lock.json")
    hashes={"complete.json":sha(folder/"complete.json"),"features.pt":sha(folder/"features.pt")}
    assert hashes["features.pt"]==cp["features"]
    packed=torch.load(folder/"features.pt",weights_only=True,map_location="cpu")
    assert packed["images"]==[r["image"] for r in rows] and packed["lock"]==cp["lock"] and packed["adapter"]==sig
    assert set(packed["images"])==set(cp["images"]) and packed["features"].shape==(448,768)
    tool_counts=Counter();tok=0
    for j,r in enumerate(rows):
        n=r["image"];p=folder/(n+".pt")
        hashes[p.name]=sha(p)
        assert hashes[p.name]==cp["images"][n]
        z=torch.load(p,weights_only=True,map_location="cpu"); fb=tools[r["rgb_sha256"]][0]
        assert (z["image"],z["query"],z["rgb"],z["feedback"],z["adapter"],z["lock"])==(n,r["orig_query"],r["rgb_sha256"],fb,sig,cp["lock"])
        assert z["actions"] and all(a and all(isinstance(t,int) for t in a) for a in z["actions"])
        assert all(x==fb for x in z["calls"])
        assert z["prior"].shape==(3584,) and torch.isfinite(z["prior"]).all()
        assert torch.equal(z["v"],packed["features"][j]) and torch.isfinite(z["v"]).all()
        tool_counts[len(z["calls"])]+=1
        tok+=sum(map(len,z["actions"]))
    return packed["features"],dict(**archive(tag,folder,hashes),adapter=sig,tool_calls_per_image=tool_counts,tokens=tok,
        env={k:cp[k] for k in ("device","cuda","torch")})

def historical(rows,bank,truth):
    out={}
    for s in "ab":
        p=FORM/"results"/("rsitmd_"+s+"_R2.pt"); z=torch.load(p,weights_only=True,map_location="cpu")
        assert z["manifest"]==sha(FORM/"inputs/rsitmd.json")
        for mode in MODES:
            ids,h,ds=counts(z["features"],bank,mode)
            v=scalar_values(h.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))
            old=truth["all_"+mode]["levels"][s+"_R2"]
            for j,k in enumerate(KEYS):
                if k in old:np.testing.assert_allclose(v[j]*100.,old[k],atol=1e-10,rtol=0)
            ids2,h2,ds2=independent(z["features"],bank,mode)
            np.testing.assert_array_equal(ids2,ids);np.testing.assert_array_equal(ds2,ds)
            np.testing.assert_allclose(h2,h,rtol=0,atol=1e-12)
            out[s+"_"+mode]=dict(images=len(ids),captions=int(ds.sum()),historical_mR=float(v[-1]),features=sha(p))
    return out

def main(only_old):
    lock=read(ROOT/"inputs/manifest.json")
    assert all(sha(ROOT/f)==h for f,h in lock["files"].items())
    assert sha(FORM/"results/rsitmd_text.pt")==lock["gallery_text"] and sha(FORM/"inputs/rsitmd.json")==lock["roster"]
    assert sha(FORM/"results/paired_results.json")==lock["historical_scores"]
    rows=read(FORM/"inputs/rsitmd.json")
    bank=torch.load(FORM/"results/rsitmd_text.pt",weights_only=True,map_location="cpu")
    old=historical(rows,bank,read(FORM/"results/paired_results.json")["rsitmd"])
    dump(ROOT/"inputs/原评分复算.json",old)
    if only_old:return
    server=read(ROOT/"inputs/server_lock.json")
    assert server["lock"]==sha(ROOT/"inputs/manifest.json")
    tools=read(FORM/"records/tools.json")
    assert sha(FORM/"records/tools.json")==lock["tool_cache"]
    values={mode:{} for mode in MODES};denom={};raw=[];assets={};result={}
    for seed in "ab":
        for arm in ("r2","r3"):
            tag=seed+"_"+arm
            visual,assets[tag]=inspect(tag,rows,tools,lock)
            for mode in MODES:
                ids,h,ds=counts(visual,bank,mode)
                i2,h2,d2=independent(visual,bank,mode)
                np.testing.assert_array_equal(ids,i2);np.testing.assert_array_equal(ds,d2)
                np.testing.assert_allclose(h,h2,rtol=0,atol=1e-12)
                assert (len(ids),ds.sum())==((446,1739) if mode=="alternatives" else (448,2240))
                assert np.array_equal(denom.setdefault(mode,ds),ds)
                values[mode][tag]=h
                raw.extend(dict(seed=seed,arm=arm,mode=mode,image=rows[i]["image"],captions=int(d),hits=x.tolist()) for i,d,x in zip(ids,ds,h))
    assert len({json.dumps(a["env"],sort_keys=True) for a in assets.values()})==1
    for mode,d in values.items():
        ds=denom[mode];block={}
        for arm in ("r2","r3","r3_minus_r2"):
            x=np.stack([d[s+"_"+arm] if arm!="r3_minus_r2" else d[s+"_r3"]-d[s+"_r2"] for s in "ab"])
            statistics=summary(x.mean(0),ds)
            for j,k in enumerate(KEYS):
                statistics[k]["seeds"]={s:float(scalar_values(z.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[j]) for s,z in zip("ab",x)}
            if arm=="r3_minus_r2":independent_change(x.mean(0),ds,statistics)
            block[arm]=statistics
        result[mode]=dict(images=len(ds),candidate_images=len(rows),captions=int(ds.sum()),**block)
    result["primary_cross_domain_gain"]=result["alternatives"]["r3_minus_r2"]["mR"]["interval"][0]>0
    dump(ROOT/"archives/index.json",assets);dump(ROOT/"results/双向检索.json",result)
    with gzip.open(ROOT/"results/逐图召回.jsonl.gz","wt",encoding="utf8") as f:
        for line in raw:f.write(json.dumps(line,ensure_ascii=False)+"\n")
    paths=["inputs/manifest.json","inputs/server_lock.json","inputs/原评分复算.json","archives/index.json",
           "results/双向检索.json","results/逐图召回.jsonl.gz",*lock["files"]]
    dump(ROOT/"results/complete.json",dict(fresh_priors=len(assets)*448,verified_raw_rows=len(raw),files={p:sha(ROOT/p) for p in paths}))

if __name__=="__main__":
    torch.set_num_threads(1)
    main("--historical" in sys.argv)
