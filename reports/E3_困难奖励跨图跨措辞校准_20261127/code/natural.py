"""Evaluate exact calibrated reward on untouched spontaneous answers and hidden captions."""
import gzip
import json
import numpy as np

from panel import ROOT,POOL,interval,read,save,sha,tensor
from reward import calibrated_scores,redistribute


def evaluate(row,g,raw,direction):
    items=g["items"]
    rewards=np.array(g["r2"])
    top=np.asarray(g["r2_top"])
    stats={k:(np.array([it[k] for it in items])>0).mean(1) for k in ("DE","FG")}
    rr=np.array([i["other_caption_rr"] for i in items])
    signals={"cal":calibrated_scores(raw["hidden"].float().numpy(),row["side"],direction),
             "old":np.array([min(i["hard_margins"]) for i in items])}
    values=[]
    for h in range(rr.shape[1]):
        metrics={}
        for label,signal in signals.items():
            ranks=redistribute(rewards,signal,rr,[j for j in range(rr.shape[1]) if j!=h])
            picked=np.flatnonzero(ranks==ranks.max())
            for key,v in {**stats,"hidden_RR":rr[:,h]}.items():
                metrics[label+"_"+key]=float(v[picked].mean()-v[top].mean())
            metrics[label+"_changed"]=float(set(picked)!=set(top))
            metrics[label+"_reward_changed"]=float(np.any(ranks!=rewards))
        for key in ("DE","FG","hidden_RR"):
            metrics["cal_minus_old_"+key]=metrics["cal_"+key]-metrics["old_"+key]
        values.append(metrics)
    return {k:float(np.mean([m[k] for m in values])) for k in values[0]}


def main():
    directions=np.load(ROOT/"inputs/directions.npz")
    data=read(ROOT/"inputs/test_panel.json")
    sums=[]
    by_seed={s:[] for s in "ab"}
    refs={s:read(POOL/"sampling"/(s+"_complete.json"))["files"] for s in "ab"}
    for r in data:
        if len(r["alt_ids"])<2:continue
        n=r["image"]
        per=[]
        for s in "ab":
            g=read(POOL/"scores"/s/(n+".json"))
            p=POOL/"sampling/raw"/s/(n+".pt")
            assert g["source"]==refs[s][n]==sha(p)
            raw=tensor(p)
            assert raw["eligible"] and g["rgb"]==r["rgb_sha256"]==raw["rgb"]
            m=evaluate(r,g,raw,directions[r["attribute"]])
            rec={k:r[k] for k in ("image","pair","cohort","attribute")}
            by_seed[s].append(dict(**rec,metrics=m))
            per.append(m)
        sums.append(dict(**rec,metrics={k:float(np.mean([m[k] for m in per])) for k in per[0]}))
    report=interval(sums)
    report["seeds"]={s:interval(rs) for s,rs in by_seed.items()}
    report["natural_pass"]=bool(report["metrics"]["cal_DE"]["ci"][0]>0 and report["metrics"]["cal_hidden_RR"]["mean"]>=0)
    with gzip.open(ROOT/"results/自然回答逐图.jsonl.gz","wt") as f:
        for s,rs in by_seed.items():
            for row in rs:f.write(json.dumps(dict(seed=s,**row))+"\n")
    save(ROOT/"results/自然回答留句.json",report)
    print(report["images"],report["metrics"],report["natural_pass"])


if __name__=="__main__":main()
