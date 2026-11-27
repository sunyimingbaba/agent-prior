"""Test a fixed action against both native R2 and image-specific joint selection."""
import argparse
from collections import Counter
import gzip
import json
import re

import numpy as np
import torch

from select_pairs import STAGE,SOURCE,ranked_pair,read,sha


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


def fixed_sequence():
    labels={r["image"]:r for r in read(SOURCE/"scoring/inputs/rows.json")}
    examples=[]
    count=Counter()
    for r in read(STAGE/"inputs/train_pairs.json")["rows"]:
        if r["kind"]!="joint":
            continue
        p=SOURCE/"sampling/raw/a"/(r["image"]+".pt")
        assert sha(p)==r["raw_sha256"]
        raw=torch.load(p,weights_only=True,map_location="cpu")
        example=dict(image=r["image"],attribute=labels[r["image"]]["attribute"],
                     true_detail=labels[r["image"]]["A"][0])
        for arm in ("r2","r3"):
            j=r[arm+"_winner"]
            ids=tuple(raw["paths"][j])
            if arm=="r3":count[ids]+=1
            example[arm]=dict(tokens=ids,text=raw["texts"][j],extra=raw["extra_texts"][j],
                              complete_answer=bool(re.search(r"<answer>\s*[^<\s][^<]*</answer>",raw["texts"][j],re.I)))
        examples.append(example)
    ids=min(count,key=lambda v:(-count[v],v))
    data=dict(fixed_tokens=ids,training_frequency=count[ids],images=examples)
    path=STAGE/"validation/训练优选原文.json"
    save(path,data)
    return ids


def analyze(seed):
    fixed=fixed_sequence()
    train=read(STAGE/"inputs/train_pairs.json")["rows"]
    excluded={(r["cohort"],r["pair"]) for r in train}
    fresh=STAGE/"validation98/fresh98"
    if seed=="b":
        rows=read(SOURCE/"scoring/inputs/rows.json")
        raw_dir=SOURCE/"sampling/raw/b"
        score_dir=SOURCE/"scores/b"
    else:
        rows=read(fresh/"inputs/rows.json")
        raw_dir=fresh/"raw/c"
        score_dir=fresh/"scores"
    values=[]
    for row in rows:
        pair=(row["cohort"],row["pair"])
        if pair in excluded or len(row["alt_ids"])<2:
            continue
        name=row["image"]
        raw=torch.load(raw_dir/(name+".pt"),weights_only=True,map_location="cpu")
        if not raw["eligible"]:
            continue
        g=read(score_dir/(name+".json"))
        i=g["r2"].index(max(g["r2"]))
        slots=[j for j,p in enumerate(raw["paths"]) if tuple(p)==fixed]
        chosen=slots[0] if slots else i
        rr=np.asarray([item["other_caption_rr"] for item in g["items"]])
        de,fg=[(np.asarray([item[k] for item in g["items"]])>0).mean(axis=1) for k in ("DE","FG")]
        d=[]
        for held in range(rr.shape[1]):
            _,joint=ranked_pair(g,[k for k in range(rr.shape[1]) if k!=held])
            j=i if joint is None else joint
            d.append([int(bool(slots)),int(chosen!=i),int(chosen==j),
                      de[chosen]-de[i],fg[chosen]-fg[i],rr[chosen,held]-rr[i,held],
                      de[chosen]-de[j],fg[chosen]-fg[j],rr[chosen,held]-rr[j,held]])
        values.append(dict(pair=pair,image=name,deltas=np.mean(d,axis=0).tolist()))
    groups=sorted({tuple(r["pair"]) for r in values})
    sizes=np.zeros((len(groups),10))
    for r in values:
        sizes[groups.index(tuple(r["pair"]))]+=[*r["deltas"],1]
    weights=np.random.default_rng(271127).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=20000)
    boot=weights@sizes
    means=sizes[:,:9].sum(axis=0)/len(values)
    cis=np.quantile(boot[:,:9]/boot[:,9,None],[.025,.975],axis=0).T
    keys=("available","different_from_r2","same_slot_as_joint","fixed_minus_r2_DE","fixed_minus_r2_FG","fixed_minus_r2_RR",
          "fixed_minus_joint_DE","fixed_minus_joint_FG","fixed_minus_joint_RR")
    result=dict(images=len(values),pairs=len(groups),available=sum(int(r["deltas"][0]) for r in values),
                effects={key:dict(mean=float(mean),interval=ci.tolist()) for key,mean,ci in zip(keys,means,cis)})
    save(STAGE/"validation"/("固定输出_"+seed+".json"),result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("seed",choices=["b","c"])
    r=analyze(p.parse_args().seed)
    print(r)
