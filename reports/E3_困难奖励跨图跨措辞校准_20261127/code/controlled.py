from collections import defaultdict
import numpy as np

from panel import ROOT,fit,interval,read,reference,save,tensor
from reward import calibrated_scores


def win(diff):
    return float(diff>0)+.5*float(diff==0)


def main():
    base=ROOT/"collection"
    direction=np.load(ROOT/"inputs/directions.npz")
    early=fit(reference(),lambda r:tensor(base/"train98/raw"/(r["image"]+".pt"))["content_states"].float().numpy())
    rows=read(ROOT/"inputs/test_panel.json")
    pairs=defaultdict(list)
    for r in rows:pairs[r["cohort"],r["pair"]].append(r)
    groups=[]
    for members in pairs.values():
        assert len(members)==2
        for idx,row in enumerate(members):
            name=row["image"]
            raw=tensor(base/"pool270/raw"/(name+".pt"))
            mate=tensor(base/"pool270/raw"/(members[1-idx]["image"]+".pt"))
            assert raw["texts"]==[mate["texts"][i] for i in (1,0,3,2)]
            g=read(base/"pool270/scores"/(name+".json"))
            scores=np.array([min(it["hard_margins"]) for it in g["items"]])
            metadata={k:row[k] for k in ("image","pair","cohort","attribute")}
            m={}
            for k in ("D","E"):
                t=0 if k=="D" else 2
                m["old_"+k]=win(scores[t]-scores[t+1])
            for method,bank,key,p,ordr in [("cal",direction,"states",raw,[0,1,2,3]),("content",early,"content_states",raw,[0,1,2,3]),
                                            ("mate",direction,"states",mate,[1,0,3,2])]:
                ss=calibrated_scores(p[key].float().numpy()[ordr],row["side"],bank[row["attribute"]])
                for j,k in enumerate(("D","E")):m[method+"_"+k]=win(ss[2*j]-ss[2*j+1])
            for key in ("old","cal","content","mate"):
                m[key+"_both"]=(m[key+"_D"]+m[key+"_E"])/2
            m["cal_minus_old"]=m["cal_both"]-m["old_both"]
            m["cal_minus_mate"]=m["cal_both"]-m["mate_both"]
            groups.append(dict(**metadata,metrics=m))
    out=interval(groups)
    ms=out["metrics"]
    out["controlled_pass"]=all(ms[k]["ci"][0]>.5 for k in ["cal_D","cal_E"]) and ms["cal_minus_old"]["ci"][0]>0
    out["by_attribute"]={k:interval([r for r in groups if r["attribute"]==k]) for k in sorted({r["attribute"] for r in groups})}
    save(ROOT/"results/受控逐图.json",groups)
    save(ROOT/"results/受控跨措辞.json",out)
    print(out["metrics"])


if __name__=="__main__":main()
