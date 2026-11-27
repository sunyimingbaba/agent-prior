"""Compute calibration and candidate group contrasts directly from original records."""
from argparse import ArgumentParser
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

OUT=Path(__file__).resolve().parent
STAGE=OUT.parent
REPORTS=STAGE.parent
POOL=next(REPORTS.glob("E3_回填后双收益*"))
REFERENCE=next(REPORTS.glob("E3_双收益回答*/validation98"))
OLD=REFERENCE/"受控真假回答"


def read(p):
    return json.loads(p.read_bytes())


def load(p):
    return torch.load(p,weights_only=True,map_location="cpu")


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def norm(x):
    x=np.array(x,dtype=np.float64)
    lens=np.sqrt((x*x).sum(axis=-1,keepdims=True))
    assert np.all(lens>0)
    return x/lens


def pool():
    train=read(REFERENCE/"fresh98/inputs/rows.json")
    raw=read(OLD/"generation.json")["files"]
    vectors=defaultdict(lambda:defaultdict(list))
    for row in train:
        name=row["image"]
        path=OLD/"raw"/(name+".pt")
        h=load(path)
        assert sha(path)==raw[name] and h["rgb"]==row["rgb_sha256"]
        assert h["texts"]==["<answer>"+w+"</answer>" for w in row["A"]]
        a,b=norm(h["states"].double().numpy())
        signed=(a-b) if row["side"]==0 else (b-a)
        vectors[row["attribute"]][row["pair"]].append(signed)
    bank={}
    for a,clusters in vectors.items():
        assert all(len(v)==2 for v in clusters.values())
        bank[a]=norm(np.mean([np.mean(v,axis=0) for v in clusters.values()],axis=0))
    stored=np.load(STAGE/"inputs/directions.npz")
    assert set(stored)==set(bank)
    error=max(float(np.max(np.abs(stored[k]-v))) for k,v in bank.items())
    assert error<1e-12,error
    reserved={(r["cohort"],r["pair"]) for r in read(POOL/"scoring/locks/training.json")["rows"]}
    avail={s:read(POOL/"sampling"/(s+"_complete.json")) for s in "ab"}
    target=read(POOL/"scoring/inputs/rows.json")
    groups=defaultdict(list)
    for r in target:groups[r["cohort"],r["pair"]].append(r)
    cohort=[]
    for k,rows in groups.items():
        assert sorted(r["side"] for r in rows)==[0,1]
        assert all(rows[0][w]==list(reversed(rows[1][w])) for w in ("A","B","D","E"))
        if k in reserved or rows[0]["attribute"] not in bank:continue
        if all(avail[s]["eligibility"][r["image"]] for s in "ab" for r in rows):cohort+=rows
    assert {r["image"] for r in read(STAGE/"inputs/test_panel.json")}=={r["image"] for r in cohort}
    assert {r["rgb_sha256"] for r in train}.isdisjoint(r["rgb_sha256"] for r in cohort)
    return bank,cohort,error


def infer(x,u,side):
    return (1 if side==0 else -1)*np.einsum("ij,j->i",norm(x.double().numpy()),u)


def reprojection(r2,q,rr,held):
    top=[i for i,r in enumerate(r2) if r==max(r2)]
    allowed=[]
    for i,s in enumerate(r2):
        if s==max(r2) or q[i]<=max(q[j] for j in top):continue
        if all(len(top)*Fraction(rr[i][k])>=sum(Fraction(rr[j][k]) for j in top)
               for k in range(len(rr[0])) if k!=held):
            allowed.append(i)
    if not allowed:return top
    members=top+allowed
    original=sorted([r2[i] for i in members])
    changed=np.asarray(r2).copy()
    for i in members:
        smaller=sum(q[j]<q[i] for j in members)
        tied=sum(q[j]==q[i] for j in members)
        changed[i]=np.mean(original[smaller:smaller+tied])
    return np.flatnonzero(changed==np.max(changed)).tolist()


def stats(rows):
    groups=sorted({k for k,_ in rows})
    fields=list(rows[0][1])
    values=np.zeros((len(groups),len(fields)+1))
    for k,r in rows:values[groups.index(k)]+=[1,*[r[f] for f in fields]]
    boot=np.random.default_rng(271127).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=20000)@values
    estimates=values.sum(axis=0)
    cis=np.percentile(boot[:,1:]/boot[:,:1],[2.5,97.5],axis=0)
    return {"images":len(rows),"pairs":len(groups),"metrics":{f:dict(mean=float(estimates[i+1]/estimates[0]),
                 ci=cis[:,i].tolist()) for i,f in enumerate(fields)}}


def natural():
    bank,cohort,error=pool()
    checks={"a":read(POOL/"scoring/locks/a_selection.json")["groups"],"b":read(POOL/"scoring/results/b_scores_sha256.json")}
    raw_sha={s:read(POOL/"sampling"/(s+"_complete.json"))["files"] for s in "ab"}
    imgrows=[]
    holdouts=0
    selected_counts=defaultdict(int)
    for r in cohort:
        name=r["image"]
        if len(r["alt_ids"])<2: continue
        metrics=defaultdict(list)
        for seed in "ab":
            p=POOL/"scores"/seed/(name+".json")
            blob=POOL/"sampling/raw"/seed/(name+".pt")
            assert sha(p)==checks[seed][name] and sha(blob)==raw_sha[seed][name]
            raw=load(blob)
            g=read(p)
            assert g["source"]==raw_sha[seed][name] and g["rgb"]==raw["rgb"]==r["rgb_sha256"]
            old=g["r2"]; top=[i for i,x in enumerate(old) if x==max(old)]
            scores=infer(raw["hidden"],bank[r["attribute"]],r["side"])
            rr=[x["other_caption_rr"] for x in g["items"]]
            fine={k:np.mean(np.array([x[k] for x in g["items"]])>0,axis=1) for k in ("DE","FG")}
            controls=[min(x["hard_margins"]) for x in g["items"]]
            for held in range(len(rr[0])):
                holdouts+=1
                obs={**fine,"rr":np.array([v[held] for v in rr])}
                for scorer,v in [("direction",scores),("native",controls)]:
                    new=reprojection(old,v,rr,held)
                    metrics[scorer+"_changed"].append(float(set(top)!=set(new)))
                    for k,a in obs.items():metrics[scorer+"_"+k].append(a[new].mean()-a[top].mean())
        imgrows.append(((r["cohort"],r["pair"]),{k:np.mean(ls) for k,ls in metrics.items()}))
    res={"directions_error":error,"image_seed_heldout_trials":holdouts,**stats(imgrows)}
    (OUT/"真实候选重算.json").write_text(json.dumps(res,indent=2)+"\n")
    print("natural",res)


def controlled():
    bank,cohort,error=pool()
    d=STAGE/"collection/pool270"
    imgrows=[]
    raw_proof=read(d/"generation.json")["files"]
    shards=[read(d/("scoring_"+str(i)+".json")) for i in (0,1)]
    score_proof={k:v for z in shards for k,v in z["files"].items()}
    for r in cohort:
        name=r["image"]
        assert sha(d/"raw"/(name+".pt"))==raw_proof[name]
        assert sha(d/"scores"/(name+".json"))==score_proof[name]
        h=load(d/"raw"/(name+".pt"))
        p=read(d/"scores"/(name+".json"))
        assert p["source"]==raw_proof[name] and p["projection"]==sha(d/"readouts"/(name+".pt"))
        assert h["rgb"]==r["rgb_sha256"]
        assert h["texts"]==["<answer>"+t+"</answer>" for k in ("D","E") for t in r[k]]
        q=infer(h["states"],bank[r["attribute"]],r["side"])
        weak=[min(x["hard_margins"]) for x in p["items"]]
        m={}
        for pos,label in ((0,"D"),(2,"E")):
            for scores,method in [(q,"calibrated"),(weak,"native")]:
                v,w=scores[pos:pos+2]
                m[method+"_"+label]=0.5 if v==w else float(v>w)
        m["gain"]=0.5*(m["calibrated_D"]-m["native_D"]+m["calibrated_E"]-m["native_E"])
        imgrows.append(((r["cohort"],r["pair"]),m))
    res=stats(imgrows)
    (OUT/"新措辞真假重算.json").write_text(json.dumps(res,indent=2)+"\n")
    print("controlled",res)


if __name__=="__main__":
    p=ArgumentParser(); p.add_argument("mode",choices=["natural","controlled"]); a=p.parse_args()
    torch.set_num_threads(1)
    natural() if a.mode=="natural" else controlled()
