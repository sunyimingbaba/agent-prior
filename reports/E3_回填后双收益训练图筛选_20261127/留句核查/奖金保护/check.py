"""Exploratory fixed guard on the actual R3 bonus after the matched training evaluation."""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
STAGE = OUT.parents[1]


def read(path):
    return json.loads(path.read_bytes())


def evaluate():
    rows = []
    for meta in read(STAGE/"scoring/inputs/rows.json"):
        group=read(STAGE/"scores/b"/(meta["image"]+".json"))
        if not group["eligible"] or len(group["alt_ids"])<2:
            continue
        items = group["items"]
        rr = np.array([it["other_caption_rr"] for it in items])
        fine = {k:(np.array([it[k] for it in items])>0).mean(axis=1) for k in ("DE","FG")}
        base = np.asarray(group["r2"])
        bonus = np.asarray(group["bonus"])
        original = group["r2_top"]
        total = {"r2":[], "r3":[]}
        for k in range(rr.shape[1]):
            visible=[sum(Fraction.from_float(float(x)) for j,x in enumerate(a) if j!=k) for a in rr]
            threshold=sum(visible[i] for i in original)
            keep=np.array([v*len(original)>=threshold for v in visible])
            guarded = base + np.where(keep,bonus,0)
            chosen=np.flatnonzero(guarded==guarded.max()).tolist()
            for model,compare in [("r2",original),("r3",group["r3_top"])]:
                difference=[z[chosen].mean()-z[compare].mean() for z in [fine["DE"],fine["FG"],rr[:,k]]]
                total[model].append([*difference,int(chosen!=compare)])
        rows.append((meta,{key:np.mean(v,axis=0) for key,v in total.items()}))
    return rows


def run():
    rows=evaluate()
    labels=sorted({(r["cohort"],r["pair"]) for r,_ in rows})
    weights=np.random.default_rng(271127).multinomial(len(labels),np.full(len(labels),1/len(labels)),size=20000)
    result=dict(images=len(rows),pairs=len(labels))
    for model in ("r2","r3"):
        arr=np.zeros((len(labels),5))
        for r,values in rows:
            arr[labels.index((r["cohort"],r["pair"]))]+=[*values[model],1]
        boot=weights@arr
        effect=arr.sum(axis=0)[:4]/arr[:,4].sum()
        ci=np.quantile(boot[:,:4]/boot[:,4,None],[.025,.975],axis=0)
        result["versus_"+model]={k:dict(mean=v,interval=interval.tolist()) for k,v,interval in
                             zip(("DE","FG","held_rr","winner_changed"),effect,ci.T)}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    run()
