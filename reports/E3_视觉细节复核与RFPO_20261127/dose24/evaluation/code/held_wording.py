"""Paired held-out wording audit on the same 56 first-caption outputs."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parents[1]/"E3_R2逐候选分值与困难细节加分_20261127/evaluation/code"))
from score_graded import RFPO, FORM, read, dump, sha, paired, summary_by_seed
from single_scores import fixed

PREVIOUS = ROOT.parents[1]/"E3_R3改选收益校准与双批首句验证_20261126"


def main():
    gallery = read(FORM/"inputs/ucm.json")
    ids = {r["rgb_sha256"]:i for i,r in enumerate(gallery)}
    known = {(r["attribute"],r["side"],*r["D"],*r["E"]) for r in read(PREVIOUS/"inputs/real_roster.json")}
    rows = [r for r in read(RFPO/"inputs/panel.json") if r["rgb_sha256"] in ids and (r["attribute"],r["side"],*r["D"],*r["E"]) in known]
    assert len(rows)==56
    fresh = torch.load(PREVIOUS/"inputs/fresh_text.pt",weights_only=True)
    assert fresh["source"] == sha(PREVIOUS/"inputs/独立措辞.json")
    extra = read(PREVIOUS/"inputs/独立措辞.json")
    words = {w:i for i,w in enumerate(fresh["words"])}
    hard = torch.load(RFPO/"records/text.pt",weights_only=True)
    full = read(ROOT/"evaluation/results/逐图指标.json")
    values={}
    for arm in ("r2","strict"):
        values[arm]={}
        for seed in "ab":
            folder=ROOT/"evaluation/records"/(seed+"_"+arm)
            assert read(folder/"complete.json")["features"]==sha(folder/"features.pt")
            z=torch.load(folder/"features.pt",weights_only=True)
            assert z["images"] == [r["image"] for r in gallery]
            v=z["features"][[ids[r["rgb_sha256"]] for r in rows]]
            e=fixed(v, fresh["features"]); d=fixed(v,hard["hard"])
            de=[];fg=[]
            for j,r in enumerate(rows):
                hr=hard["maps"]["primary"][r["image"]];side=r["side"]
                de.append([d[j,hr[k][0]]-d[j,hr[k][1]] for k in "DE"])
                fg.append([e[j,words[w[side]]]-e[j,words[w[1-side]]] for w in extra[r["attribute"]]])
            result={}
            for name,mat in (("DE",de),("FG",fg)):
                result[name]=(np.array(mat)>0).mean(1)
                result[name+"_margin"]=np.array(mat).mean(1)
            ref={x["image"]:x["first106"]["DE"] for x in full if x["arm"]==arm and x["seed"]==seed and "first106" in x}
            np.testing.assert_array_equal(result["DE"],[ref[r["image"]] for r in rows])
            values[arm][seed]=result
    labels=[r["pair"] for r in rows]
    out={"images":len(rows),"pairs":len(set(labels)),
         "levels":{k:summary_by_seed(v,labels) for k,v in values.items()},
         "strict_minus_r2":paired(values["strict"],values["r2"],labels)}
    dump(ROOT/"evaluation/results/单次新措辞复核.json",out)


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
