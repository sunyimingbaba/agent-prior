"""Independent scalar replay of image selection, pair orientation and held-caption diagnostics."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/"E3_回填后双收益训练图筛选_20261127"
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def choices(g,cols,anchor=False):
    items=g["items"]
    rewards=g["r2"]
    gold=[sum(m>0 for m in o["hard_margins"]) for o in items]
    i=rewards.index(max(rewards))
    candidates=[]
    for j,(r,item) in enumerate(zip(rewards,items)):
        if r>=rewards[i]:
            continue
        changed=[item["other_caption_rr"][c]-items[i]["other_caption_rr"][c] for c in cols]
        if anchor:
            if gold[j]<=gold[i] and all(x<=0 for x in changed):
                candidates.append(j)
        elif gold[j]>gold[i] and all(x>=0 for x in changed):
            candidates.append(j)
    if not candidates:
        return i,None
    if anchor:
        return i,min(candidates,key=lambda j:(rewards[j],gold[j],j))
    return i,max(candidates,key=lambda j:(gold[j],min(items[j]["hard_margins"]),rewards[j],-j))


def source(seed):
    sign=(read(OLD/"scoring/locks/a_selection.json")["groups"] if seed=="a"
          else read(OLD/"scoring/results/b_scores_sha256.json"))
    groups={}
    for name,h in sign.items():
        f=OLD/"scores"/seed/(name+".json")
        assert sha(f)==h
        groups[name]=read(f)
    return groups


def select(rows,a):
    pool=sorted([r for r in rows if a[r["image"]]["eligible"] and len(r["alt_ids"])>=2],
                key=lambda r:r["rgb_sha256"])
    eligible=defaultdict(dict)
    for r in pool:
        g=a[r["image"]]
        for kind in ("joint","anchor"):
            i,j=choices(g,range(len(r["alt_ids"])),anchor=kind=="anchor")
            if j is not None:
                eligible[kind][r["image"]]=(i,j)
    joint=list(eligible["joint"])[:12]
    anchor=[n for n in eligible["anchor"] if n not in joint][:len(joint)]
    joint=joint[:len(anchor)]
    raw_sign=read(OLD/"sampling/a_complete.json")["files"]
    records=[]
    for anc,good in zip(anchor,joint):
        for kind,n in (("anchor",anc),("joint",good)):
            i,j=eligible[kind][n]
            r=next(r for r in rows if r["image"]==n)
            path=OLD/"sampling/raw/a"/(n+".pt")
            assert sha(path)==raw_sign[n]
            raw=torch.load(path,weights_only=True)
            assert raw["paths"][i]!=raw["paths"][j]
            pos,neg=(i,j) if kind=="anchor" else (j,i)
            records.append(dict(image=n,rgb=r["rgb_sha256"],cohort=r["cohort"],pair=r["pair"],kind=kind,
                                baseline=i,alternative=j,preferred=pos,rejected=neg,raw_a=raw_sign[n],
                                preferred_tokens=len(raw["paths"][pos]),rejected_tokens=len(raw["paths"][neg]),
                                preferred_continuation=len(raw["extras"][pos]),rejected_continuation=len(raw["extras"][neg])))
    return dict(pool=len(pool),joint_nodes=len(eligible["joint"]),anchor_nodes=len(eligible["anchor"]),
                ordered=records)


def held(rows,b,schedule):
    removed={(z["cohort"],z["pair"]) for z in schedule["ordered"]}
    out=[]
    for r in rows:
        name=r["image"]; key=(r["cohort"],r["pair"]);g=b[name]
        if key in removed or not g["eligible"] or len(g["alt_ids"])<2:
            continue
        folds=[]
        for omit in range(len(g["alt_ids"])):
            i,j=choices(g,[k for k in range(len(g["alt_ids"])) if k!=omit])
            chosen=i if j is None else j
            old,new=g["items"][i],g["items"][chosen]
            f=dict(hidden=omit,i=i,chosen=chosen,switched=int(j is not None),
                   MRR=new["other_caption_rr"][omit]-old["other_caption_rr"][omit])
            for label in ("DE","FG"):
                if new[label] is not None:
                    f[label]=(sum(x>0 for x in new[label])-sum(x>0 for x in old[label]))/2
            folds.append(f)
        out.append(dict(image=name,pair=list(key),folds=folds))
    return out


def summarize(obs):
    pairs=sorted({tuple(r["pair"]) for r in obs});ind={pair:i for i,pair in enumerate(pairs)}
    weights=np.random.default_rng(271127).multinomial(len(pairs),np.repeat(1/len(pairs),len(pairs)),size=20000)
    summary={"images":len(obs),"pairs":len(pairs),"folds":sum(len(x["folds"]) for x in obs),"metrics":{}}
    for metric in ("switched","MRR","DE","FG"):
        base=np.zeros((len(pairs),2))
        for r in obs:
            vals=[f[metric] for f in r["folds"] if metric in f]
            if vals:
                base[ind[tuple(r["pair"])]] += [np.mean(vals),1]
        draw=weights@base
        summary["metrics"][metric]=dict(value=float(base[:,0].sum()/base[:,1].sum()),
                            interval=np.quantile(draw[:,0]/draw[:,1],[.025,.975]).tolist())
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("a","b"))
    args=p.parse_args()
    rows=read(OLD/"scoring/inputs/rows.json")
    if args.phase=="a":
        sel=select(rows,source("a"));save(HERE/"独立A选对.json",sel)
        print(sel["pool"],sel["joint_nodes"],sel["anchor_nodes"],len(sel["ordered"]))
    else:
        sel=read(HERE/"独立A选对.json")
        obs=held(rows,source("b"),sel)
        result=summarize(obs)
        save(HERE/"独立B留句.json",result)
        print(result)
