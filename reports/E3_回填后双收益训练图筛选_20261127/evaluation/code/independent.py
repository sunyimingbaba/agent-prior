"""Recompute the three formal endpoints from image features and the frozen text banks."""
import argparse
from collections import defaultdict
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import torch

EVAL=Path(__file__).resolve().parents[1]
REPORTS=EVAL.parents[1]
PREV=REPORTS/"E3_R2逐候选分值与困难细节加分_20261127/evaluation"
FORM=REPORTS/"E3-2_UCM与RSITMD独立复验_20261124"
FINE=REPORTS/"E3_UCM多描述奖励同预算RFPO_20261124"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.load(path,map_location="cpu",weights_only=True)


def hits(values, positives):
    best=values[positives].max()
    over=np.count_nonzero(values>best)
    equal=np.count_nonzero(values==best)
    good=np.count_nonzero(values[positives]==best)
    def recall(k):
        left=k-over
        if left<=0:
            return 0.
        if left>equal-good:
            return 1.
        return 1.-comb(equal-good,left)/comb(equal,left)
    return np.array([recall(k) for k in (1,5,10)])


def independently_score(v, gallery, fine_rows, bank, hard):
    # Every caption owner occupies five columns even where text tokens repeat.
    by_caption=bank["caption_order"].numpy().reshape(-1,5)
    gallery_ids={r["image"]:i for i,r in enumerate(gallery)}
    sim=v@bank["native"].double().numpy().T
    ordered=sim[:,by_caption.ravel()]
    i2t=np.stack([hits(z,np.arange(i*5,i*5+5)) for i,z in enumerate(ordered)])
    t2i=np.stack([hits(z,[j//5]) for j,z in enumerate(ordered.T)]).reshape(210,5,3).mean(1)
    result={"mR":np.concatenate([i2t,t2i],axis=1).mean(1)}
    detail=v@hard["hard"].double().numpy().T
    de,mrr=[],[]
    for row in fine_rows:
        name=row["image"]
        i=gallery_ids[name]
        ix=hard["maps"]["primary"][name]
        de.append(np.mean([int(detail[i,ix[l][0]]>detail[i,ix[l][1]]) for l in "DE"]))
        ids=by_caption[i]
        other=set(ids)-{ids[0]}
        neg=sim[i,~np.isin(np.arange(sim.shape[1]),ids)]
        rr=[]
        for j in sorted(other):
            val=sim[i,j]
            start=1+np.count_nonzero(neg>val)
            end=start+np.count_nonzero(neg==val)
            rr.append(sum(1/k for k in range(start,end+1))/(end-start+1))
        assert rr
        mrr.append(np.mean(rr))
    return {**result,"DE":np.array(de),"mrr":np.array(mrr)}


def collect(folder,gallery,fine,bank,hard):
    raw_path=folder/"results/逐图指标.json"
    lock=read(folder/"results/complete.json")["files"]
    assert sha(raw_path)==lock["results/逐图指标.json"]
    rows={(r["seed"],r["arm"],r["image"]):r for r in read(raw_path)}
    data=defaultdict(list)
    max_error=0.
    feature_sources={}
    for arm in ["r2","r3"]:
        for s in "ab":
            tag=f"{s}_{arm}"
            feature=folder/"records"/tag/"features.pt"
            source=sha(feature)
            assert source==read(feature.with_name("complete.json"))["features"]
            feature_sources[tag]=source
            z=load(feature)
            assert z["images"]==[r["image"] for r in gallery] and z["features"].shape==(210,768)
            raw=independently_score(z["features"].double().numpy(),gallery,fine,bank,hard)
            for key,obs in raw.items():
                block="official" if key=="mR" else "first106"
                order=gallery if key=="mR" else fine
                theirs=[rows[s,arm,r["image"]][block][key] for r in order]
                error=float(np.abs(obs-theirs).max())
                max_error=max(max_error,error)
                assert error<1e-12,(tag,key,error)
                data[arm,key].append(obs)
    return {k:np.stack(x) for k,x in data.items()},dict(features=feature_sources,
             image_metrics_sha256=sha(raw_path),max_abs_per_image_error=max_error)


def summarize(values,units):
    # Two seeds remain paired within each image. Predefined units are equal-sized
    # (one image for mR, two images in each of the 53 fine-detail pairs).
    groups=list(dict.fromkeys(units))
    groups_idx=[[i for i,u in enumerate(units) if u==g] for g in groups]
    assert len({len(idx) for idx in groups_idx})==1
    group_means=np.stack([values[:,idx].mean(1) for idx in groups_idx],axis=1)
    sample=np.random.default_rng(20261125).integers(len(groups),size=(20000,len(groups)))
    def level(z):
        return dict(mean=float(z.mean()),interval=np.quantile(z[sample].mean(1),[.025,.975]).tolist())
    return dict(**level(group_means.mean(0)),seeds={k:level(group_means[j]) for j,k in enumerate("ab")},
                images=len(units),units=len(groups))


def main(previous_only):
    gallery=read(FORM/"inputs/ucm.json")
    by_rgb={r["rgb_sha256"]:r["image"] for r in gallery}
    fine=[r for r in read(FINE/"inputs/panel.json") if r["rgb_sha256"] in by_rgb]
    assert len(gallery)==210 and len(fine)==106 and len(set(r["pair"] for r in fine))==53
    assert all(by_rgb[r["rgb_sha256"]]==r["image"] for r in fine)
    bank=load(FORM/"results/ucm_text.pt")
    hard=load(FINE/"records/text.pt")
    previous,ps=collect(PREV,gallery,fine,bank,hard)
    current,cs=(previous,ps) if previous_only else collect(EVAL,gallery,fine,bank,hard)
    folder=PREV if previous_only else EVAL
    official=read(folder/"results/单次检索对照.json")
    out={"independent_code_sha256":sha(Path(__file__)),"bootstrap_repetitions":20000,"bootstrap_seed":20261125,"feature_checks":{"previous":ps,"current":cs},
         "text_banks":{"native":sha(FORM/"results/ucm_text.pt"),"hard":sha(FINE/"records/text.pt")},"metrics":{}}
    for key in ("DE","mrr","mR"):
        units=[r["image"] for r in gallery] if key=="mR" else [r["pair"] for r in fine]
        a,b=current["r3",key],current["r2",key]
        pa,pb=previous["r3",key],previous["r2",key]
        parts={"screened_r3":a,"screened_r2":b,"screened_r3_minus_r2":a-b,"previous_r3":pa,
               "previous_r2":pb,"previous_r3_minus_r2":pa-pb,"screened_minus_previous_r3_descriptive":a-pa,
               "screened_minus_previous_r2_descriptive":b-pb,"change_in_arm_contrast_descriptive":(a-b)-(pa-pb)}
        values={name:summarize(vec,units) for name,vec in parts.items()}
        actual=values["screened_r3_minus_r2"]
        expected=official["official" if key=="mR" else "first106"]["contrasts"]["r3-r2"][key]
        assert np.allclose([actual["mean"],*actual["interval"],*[v["mean"] for v in actual["seeds"].values()]],
                           [expected["mean"],*expected["interval"],*expected["seeds"]],rtol=0,atol=1e-12)
        out["metrics"][key]=values
    get=lambda k:out["metrics"][k]["screened_r3_minus_r2"]
    out["criteria"]=dict(detail_lower_positive=get("DE")["interval"][0]>0,mR_preserved=get("mR")["mean"]>=0,
                         other_caption_MRR_preserved=get("mrr")["mean"]>=0)
    out["passes"]=all(out["criteria"].values())
    assert out["criteria"]==official["criteria"]
    if previous_only:
        print(out["criteria"],ps)
    else:
        p=EVAL/"results/独立验算.json"
        p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
        print(out["criteria"],cs)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--previous-only",action="store_true")
    main(parser.parse_args().previous_only)
