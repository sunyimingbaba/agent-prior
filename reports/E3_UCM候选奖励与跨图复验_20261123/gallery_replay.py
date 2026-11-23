"""Apply locked teacher choices to the original 210-image gallery and a caption-disjoint evaluation."""
import argparse
import hashlib

import numpy as np

from experiment import CROSS, FRESH, OUT, REPORTS, METRICS, contrast, datasets, read, rows, save, sha, table, tensor


def success(scores, labels):
    top = []
    for k in (1,5,10):
        threshold=np.partition(scores,-k,axis=1)[:,-k,None]
        higher=scores>threshold
        equal=scores==threshold
        guaranteed=(higher & labels).any(axis=1)
        tie=equal.sum(axis=1)
        positive=(equal & labels).sum(axis=1)
        slots=k-higher.sum(axis=1)
        miss=np.ones(len(scores))
        for i in range(k):
            active=slots>i
            miss[active]*=np.maximum(tie[active]-positive[active]-i,0)/(tie[active]-i)
        top.append(np.where(guaranteed,1,1-miss))
    return np.stack(top,axis=1)


def per_source(scores, keep, owner):
    target=(np.arange(len(scores))[:,None]==owner[None,:])
    mask=target[:,keep].any(axis=1)
    sc=scores[:,keep]
    i2t=success(sc[mask],target[:,keep][mask])
    t2i=success(sc.T,target[:,keep].T)
    values=np.zeros((len(scores),6))
    values[mask,:3]=i2t
    np.add.at(values[:,3:],owner[keep],t2i)
    denom=np.column_stack([np.repeat(mask[:,None],3,axis=1),
                           np.repeat(np.bincount(owner[keep],minlength=len(scores))[:,None],3,axis=1)])
    return values,denom


def feature_scores(visual,text):
    out=np.zeros((len(visual),len(text)))
    for j in range(visual.shape[1]):
        out+=visual[:,j,None]*text[None,:,j]
    return out


def main(extra=False):
    gallery=tensor(FRESH/"gallery/native.pt").double().numpy()
    texts=tensor(FRESH/"gallery/text.pt")
    bank=texts["native"].double().numpy()
    order=np.array(texts["caption_order"])
    owner=np.repeat(np.arange(210),5)
    archive=feature_scores(gallery,bank)[:,order]
    names=read(CROSS/"val/index.json")["names"]
    main_rows=[r for r in rows(OUT/"验证逐图.csv") if r["swap"]=="0"]
    candidate_rows=[r for r in rows(OUT/"验证分项.csv") if r["swap"]=="0" and r["positives"]=="1"]
    restored=rows(OUT/"原奖励补齐14图.csv")
    for image in sorted({r["image"] for r in restored}):
        g=sorted([r for r in restored if r["image"]==image],key=lambda x:int(x["k"]))
        scores=np.array([float(r["old"]) for r in g])
        ix=np.flatnonzero(scores==scores.max())
        candidates=sorted([r for r in candidate_rows if r["image"]==image],key=lambda x:int(x["k"]))
        main_rows.append(dict(dataset="val14",image=image,cluster=image,method="old",k=",".join(map(str,ix)),
                              **{m:np.mean([float(candidates[i][m]) for i in ix]) for m in METRICS}))
    if extra:
        main_rows.extend(rows(OUT/"额外16图选择.csv"))
    for r in main_rows:
        for m in METRICS:
            r[m]=float(r[m])
    lookup={(r["image"],r["method"]):list(map(int,r["k"].split(","))) for r in main_rows}
    photos=sorted({r["image"] for r in main_rows})
    assert len(photos)==(57 if extra else 41)
    vectors={}
    for _,image,_,v,*_ in datasets("val"):
        vectors[image]=feature_scores(v,bank)[:,order]
    if extra:
        package=tensor(REPORTS/"E3-2_UCM同起点三奖励短训_20261120/eval/base/vectors.pt")["detail"]
        for image in set(photos)-set(vectors):
            ids=[package["order"].index((image,"upright",k)) for k in range(4)]
            vectors[image]=feature_scores(package["visual"][ids].double().numpy(),bank)[:,order]
    greedy={}
    for folder in [CROSS/"val",CROSS/"object"]:
        recs=rows(folder/"candidates.csv")
        features=tensor(folder/"features.pt")["visual"]
        for i,r in enumerate(recs):
            if r["mode"]=="scene" and int(r["k"])==-1:
                greedy[r["image"]]=features[i]
    for r in read(FRESH/"inputs/fresh.json"):
        f=FRESH/"outputs/fresh"/(r["image"].split(".")[0]+".pt")
        meta=read(f.with_suffix(".json"))
        slots=[i for i,z in enumerate(meta["logs"]) if (z["cue"],z["view"],z["slot"])==("scene",0,0)]
        assert len(slots)==1
        greedy[r["image"]]=tensor(f)["visual"][0,slots[0]]
    if extra:
        for im,view,k in package["order"]:
            if view=="upright" and k==-1:
                greedy[im]=package["visual"][package["order"].index((im,view,k))]
    assert set(greedy)==set(photos)
    for im in photos:
        candidate_count=len(vectors[im])
        vectors[im]=np.concatenate([vectors[im],feature_scores(greedy[im][None].double().numpy(),bank)[:,order]])
        lookup[im,"greedy"]=[candidate_count]
    filters={"full":np.ones(1050,bool)}
    forbidden=set(order.reshape(210,5)[:,0])
    for image in photos:
        positive=list(dict.fromkeys(order.reshape(210,5)[names.index(image)]))
        forbidden.add(positive[1])
    odd=np.array([int(hashlib.sha256(",".join(map(str,tk)).encode()).hexdigest()[-1],16)%2==1
                  for tk in texts["native_tokens"].tolist()])
    filters["unused"]=odd[order] & ~np.isin(order,list(forbidden))
    draws=np.random.default_rng(20261123).random((64,len(photos)))
    all_values={}
    details=[]
    methods=sorted({r["method"] for r in main_rows})
    for method in ["archive","greedy"]+methods:
        ids=np.array([names.index(im) for im in photos])
        if method=="archive":
            schedules=[None]
        else:
            schedules=[tuple(lookup[im,method][int(x*len(lookup[im,method]))] for im,x in zip(photos,draw,strict=True))
                       for draw in draws]
            if len(set(schedules)) == 1:
                schedules=schedules[:1]
        sums={}
        cache={}
        for schedule in schedules:
            if schedule not in cache:
                sc=archive.copy()
                if schedule is not None:
                    sc[ids]=np.stack([vectors[im][k] for im,k in zip(photos,schedule,strict=True)])
                cache[schedule]={label:per_source(sc,keep,owner) for label,keep in filters.items()}
            for label,(v,den) in cache[schedule].items():
                if label not in sums:
                    sums[label]=np.zeros_like(v)
                sums[label]+=v
        for label,summed in sums.items():
            v=summed/len(schedules)
            den=cache[schedules[0]][label][1]
            rr=v.sum(0)/den.sum(0)
            all_values[method,label]=(v,den)
            details.append(dict(method=method,gallery=label,images=int(den[:,0].sum()),captions=int(den[:,3].sum()),
                                mR=rr.mean(),**dict(zip(["i1","i5","i10","t1","t5","t10"],rr,strict=True))))
    assert abs(next(r["mR"] for r in details if r["method"]=="archive" and r["gallery"]=="full")-.6006)<.0001
    prefix="全部57图" if extra else ""
    table(prefix+"整库召回.csv",details)
    stats={}
    bs=np.random.default_rng(20261123).multinomial(210,np.full(210,1/210),size=20000)
    for m in methods:
        if m in ["old","random"]:continue
        for ref in ["old","random","p1_tol0","greedy"]:
            if m==ref:continue
            dif={}
            for label in filters:
                v,den=all_values[m,label]
                delta=v-all_values[ref,label][0]
                conf=((bs@delta)/(bs@den)).mean(1)
                dif[label]=dict(avg=(delta.sum(0)/den.sum(0)).mean(),lo=np.quantile(conf,.025),hi=np.quantile(conf,.975))
            if ref!="greedy":
                dif["offline"]=contrast(main_rows,m,ref)
            stats[m+" vs "+ref]=dif
    save(prefix+"整库配对差.json",dict(contrasts=stats,draws=64,old_complete=True,proxy=read(OUT/"原奖励复核.json"),
                               protocol=sha(OUT/"整库复验口径.md"),code=sha(OUT/"gallery_replay.py")))
    print(details)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--extra",action="store_true")
    main(parser.parse_args().extra)
