"""Group-cross-fitted ridge transfer between direction-phrase difference vectors."""
import json

import numpy as np
from scipy.stats import rankdata

from experiment import CROSS, FRESH, INITIAL, OUT, averages, best, contrast, datasets, read, rows, save, table, tensor


def unit(v):
    return v / np.linalg.norm(v)


def diff(bank, phrases):
    return unit(bank[phrases[0]]-bank[phrases[1]])


def direction_texts():
    src={}
    dest={}
    cases={}
    t=tensor(INITIAL/"scoring/vectors.pt")
    bank=dict(zip(t["texts"],t["text"].double().numpy(),strict=True))
    cbank=tensor(INITIAL/"heldout/text.pt")["features"].double().numpy()
    ci=read(INITIAL/"heldout/index.json")
    for rec in read(INITIAL/"inputs/selected.json"):
        im=rec["image"]
        src[im]=diff(bank,rec["A"])
        a,b=ci["columns"][im]
        dest[im]=(diff(bank,rec["B"])+unit(cbank[a]-cbank[b]))/2
        cases[im]=rec["block"]
    for split in ["train","val"]:
        t=tensor(CROSS/split/"features.pt")
        info=read(CROSS/split/"index.json")
        bank=dict(zip(info["texts"],t["text"].double().numpy(),strict=True))
        for rec in read(CROSS/"inputs/retained.json"):
            if rec["split"]!=split:continue
            im=rec["image"]
            src[im]=diff(bank,rec["A"])
            if split=="train":
                dest[im]=(diff(bank,rec["C"])+diff(bank,rec["D"]))/2
            cases[im]=rec["block"]
    t=tensor(CROSS/"object/features.pt")
    bank=dict(zip(t["texts"],t["text"].double().numpy(),strict=True))
    for rec in read(CROSS/"object_inputs/cases.json"):
        if rec["A"][0] in bank:
            src[rec["image"]]=diff(bank,rec["A"])
    for rec in read(FRESH/"inputs/fresh.json"):
        t=tensor(FRESH/"outputs/fresh"/(rec["image"].split(".")[0]+".pt"))["text"].double().numpy()
        src[rec["image"]]=unit(t[0]-t[1])
    return src,dest,cases


def compute(phase, alphas, weights, src,dest,cases):
    raw=rows(OUT/("开发分项.csv" if phase=="train" else "验证分项.csv"))
    groups={}
    for r in raw:
        if r["swap"]=="0" and r["positives"]=="1":
            groups.setdefault(r["image"],[]).append(r)
    train_ids=set(read(OUT/"开发锁定.json")["images"])
    out=[]
    for ds,im,cluster,v,_,held,_ in datasets(phase):
        if im not in groups:continue
        data=sorted(groups[im],key=lambda r:int(r["k"]))
        captions=np.array([float(r["score"]) for r in data])
        scores={k:np.array([float(r[k]) for r in data]) for k in ("rr","native_margin","all_rr","rr_first_excluded")}
        scores.update(direction=(held>0).mean(1),direction_margin=held.mean(1))
        ids=sorted(j for j in train_ids if phase!="train" or cases[j]!=cases[im])
        x=np.stack([src[j] for j in ids])
        y=np.stack([dest[j] for j in ids])
        k=x@x.T
        selectors={"caption":best(captions)}
        for weight in weights:
            selectors[f"A_{weight}"]=best((100-weight)*rankdata(captions)+weight*rankdata(v@src[im]))
        for alpha in alphas:
            projected=src[im]+(src[im]@x.T)@np.linalg.solve(k+alpha*np.eye(len(x)),y-x)
            for weight in weights:
                selectors[f"transfer_{alpha:g}_{weight}"]=best((100-weight)*rankdata(captions)+weight*rankdata(v@projected))
        for method,idx in selectors.items():
            out.append(dict(dataset=ds,image=im,cluster=cluster,method=method,k=",".join(map(str,idx)),
                            **{k:float(z[idx].mean()) for k,z in scores.items()}))
    return out


def main():
    src,dest,cases=direction_texts()
    alphas=[.1,1,10]
    weights=[25,50]
    dev=compute("train",alphas,weights,src,dest,cases)
    table("跨措辞拟合逐图开发.csv",dev)
    av=averages(dev)
    eligible=[]
    for alpha in alphas:
        for w in weights:
            name=f"transfer_{alpha:g}_{w}"
            d=av[name]["direction"]-av[f"A_{w}"]["direction"]
            if d>0 and av[name]["direction"]>av["caption"]["direction"] and av[name]["rr"]>=av["caption"]["rr"]-.02:
                eligible.append((d,av[name]["rr"],alpha,w))
    save("跨措辞拟合开发.json",dict(means=av,eligible=eligible))
    print(av,eligible)
    if eligible:
        _,_,alpha,w=max(eligible)
        val=compute("val",[alpha],[w],src,dest,cases)
        name=f"transfer_{alpha:g}_{w}"
        table("跨措辞拟合逐图复验.csv",val)
        save("跨措辞拟合复验.json",dict(means=averages(val),vs_A=contrast(val,name,f"A_{w}"),vs_caption=contrast(val,name,"caption")))


if __name__=="__main__":
    main()
