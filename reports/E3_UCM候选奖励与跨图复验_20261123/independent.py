"""Confirm unchanged selectors on a fourth, disjoint, 4-candidate image panel."""
import hashlib

import numpy as np
from scipy.stats import rankdata

from experiment import CROSS, FRESH, METRICS, OUT, REPORTS, averages, best, contrast, load_gallery, read, rows, save, select_close, sha, table, tensor
from proxy_replay import scorer, total_reward


def main():
    prev=REPORTS/"E3-2_UCM同起点三奖励短训_20261120"
    path=prev/"eval/base"
    proof=read(path/"scored.json")
    assert sha(path/"vectors.pt")==proof["vectors"] and sha(path/"text.pt")==proof["text"]
    data=tensor(path/"vectors.pt")["detail"]
    text=tensor(path/"text.pt")
    textvec=dict(zip(text["texts"],text["relation"].double().numpy(),strict=True))
    obj=scorer()
    names=read(CROSS/"val/index.json")["names"]
    gallery,even,positive=load_gallery("val")
    assert np.allclose(gallery,text["native"].double().numpy(),rtol=0,atol=2e-5)
    oldimages={r["image"] for r in rows(OUT/"验证逐图.csv")} | set(read(OUT/"开发锁定.json")["images"])
    result=[]
    details=[]
    exclusions=[]
    for rec in read(prev/"inputs/detail.json"):
        im=rec["image"]
        assert im not in oldimages
        keys=[(im,"upright",k) for k in range(4)]
        ids=[data["order"].index(key) for key in keys]
        v=data["visual"][ids].double().numpy()
        prefix=path/"detail"/(im.split(".")[0]+"_upright")
        h=tensor(prefix.with_suffix(".pt"))
        meta=read(prefix.with_suffix(".json"))
        assert sha(prefix.with_suffix(".pt"))==meta["sha256"]
        ranks=[]
        old=[]
        for k in range(4):
            tr=meta["traces"][k+1]
            assert tr["k"]==k
            rank=obj.score(h[k+1].flatten().float().tolist(),names.index(im))[0]
            ranks.append(rank)
            old.append(total_reward("\n".join(q["text"] for q in tr["rounds"]),rank))
        def direction(phrase):
            a,b=rec[phrase]
            return v @ (textvec[a]-textvec[b])
        reward_dir=direction("C")
        test_dir=direction("D")
        ps=positive[im]
        sims=v@gallery.T
        neg=np.ones(len(even),bool)
        neg[ps]=False
        n0=sims[:,neg & even].max(1)
        n1=sims[:,neg & ~even]
        a=sims[:,ps[0]]-n0
        b=sims[:,ps[:2]].mean(1)-n0
        selectors=dict(old=best(np.array(old)), random=np.arange(4), p1_tol0=best(a), p2_tol0=best(b),
                       fixed25=best(3*rankdata(a)+rankdata(reward_dir)))
        for n,x in [(1,a),(2,b)]:
            tol=read(OUT/"开发锁定.json")["thresholds"][str(n)]
            selectors[f"p{n}_tol{tol:g}"]=select_close(x,reward_dir,tol)
        metric=dict(direction=(test_dir>0).astype(float),direction_margin=test_dir)
        if len(ps)<3:
            exclusions.append(im)
        else:
            pos=sims[:,ps[2:]]
            rr=1/(1+(n1[:,:,None]>pos[:,None,:]).sum(1))
            metric.update(rr=rr.max(1),native_margin=pos.max(1)-n1.max(1),all_rr=rr.mean(1),
                          rr_first_excluded=1/(1+(n1>sims[:,ps[1:]].max(1)[:,None]).sum(1)))
        for k in range(4):
            details.append(dict(image=im,k=k,rank0=ranks[k],old=old[k],caption=a[k],reward_C=reward_dir[k],held_D=test_dir[k]))
        for m,inds in selectors.items():
            result.append(dict(dataset="extra16",image=im,cluster=im,method=m,k=",".join(map(str,inds)),
                               **{key:float(metric[key][inds].mean()) if key in metric else None for key in METRICS}))
    table("额外16图候选.csv",details)
    table("额外16图选择.csv",result)
    old_rows=[r for r in rows(OUT/"验证逐图.csv") if r["swap"]=="0"]
    old_scores=rows(OUT/"原奖励补齐14图.csv")
    raw=[r for r in rows(OUT/"验证分项.csv") if r["positives"]=="1" and r["swap"]=="0"]
    for im in {r["image"] for r in old_scores}:
        s=sorted([r for r in old_scores if r["image"]==im],key=lambda r:int(r["k"]))
        ids=best(np.array([float(r["old"]) for r in s]))
        q=sorted([r for r in raw if r["image"]==im],key=lambda r:int(r["k"]))
        old_rows.append(dict(method="old",image=im,cluster=im,dataset="val14",
                             **{m:np.mean([float(q[k][m]) for k in ids]) for m in METRICS}))
    for r in old_rows:
        for m in METRICS:r[m]=float(r[m])
    summary={}
    for name,g in [("extra16",result),("combined",result+old_rows)]:
        eligible=[r for r in g if r["rr"] is not None]
        dif={}
        for m,ref in [("p1_tol0","old"),("p1_tol0","random"),("fixed25","p1_tol0"),("p1_tol0.001","p1_tol0"),
                      ("p2_tol0.001","p2_tol0")]:
            dif[m+" vs "+ref]=contrast(eligible,m,ref)
        summary[name]=dict(means=averages(eligible),diff=dif,
                           direction_all={m:float(np.mean([r["direction"] for r in g if r["method"]==m]))
                                          for m in {r["method"] for r in g}})
    save("额外复验.json",dict(excluded_native=exclusions,groups=summary,code=sha(OUT/"independent.py"),
                           protocol=sha(OUT/"额外16图口径.md")))
    print("Fourth panel complete",exclusions)


if __name__=="__main__":
    main()
