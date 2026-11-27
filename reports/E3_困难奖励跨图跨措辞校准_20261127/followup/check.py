"""Test a single frozen veto against the same original readout without the veto."""
from collections import defaultdict
from fractions import Fraction
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from panel import read,save,tensor,interval
from reward import calibrated_scores,redistribute


def main():
    folder=ROOT/"followup_pool"
    dirs=np.load(ROOT/"inputs/directions.npz")
    rows=read(folder/"inputs/rows.json")
    out=[]
    for row in rows:
        name=row["image"]
        raw=tensor(folder/"raw/d"/(name+".pt"))
        if not raw["eligible"]:
            continue
        group=read(folder/"scores"/(name+".json"))
        its=group["items"]
        rr=np.array([it["other_caption_rr"] for it in its])
        if rr.shape[1]<2:
            continue
        base=np.array(group["r2"])
        scores=calibrated_scores(raw["hidden"].float().numpy(),row["side"],dirs[row["attribute"]])
        weak=np.array([min(it["hard_margins"]) for it in its])
        correct=np.array([sum(v>0 for v in it["hard_margins"]) for it in its])
        fine={k:(np.array([it[k] for it in its])>0).mean(1) for k in ("DE","FG") if its[0].get(k)}
        top=np.flatnonzero(base==base.max())
        valid=(correct*len(top)>correct[top].sum())
        # Disallowed lower-ranked candidates cannot join the redistribution.
        strict=np.where(valid|(base==base.max()),weak,min(weak))
        veto=np.where(((scores>max(scores[top]))&valid)|(base==base.max()),weak,min(weak))
        data=defaultdict(list)
        for h in range(rr.shape[1]):
            visible=[k for k in range(rr.shape[1]) if k!=h]
            for arm,signal in [("cal",scores),("old",weak),("strict",strict),("veto",veto)]:
                result=redistribute(base,signal,rr,visible)
                winner=np.flatnonzero(result==result.max())
                for k,measure in dict(**fine,hidden_RR=rr[:,h]).items():
                    data[arm+"_"+k].append(float(measure[winner].mean()-measure[top].mean()))
                data[arm+"_changed"].append(float(set(winner)!=set(top)))
        metrics={k:np.mean(v) for k,v in data.items()}
        for k in [*fine,"hidden_RR","changed"]:
            metrics["veto_minus_strict_"+k]=metrics["veto_"+k]-metrics["strict_"+k]
            metrics["cal_minus_old_"+k]=metrics["cal_"+k]-metrics["old_"+k]
        out.append(dict(image=name,pair=row["pair"],cohort="followup",attribute=row["attribute"],metrics=metrics))
    summary=interval(out)
    save(ROOT/"followup/逐图.json",out)
    save(ROOT/"followup/独立复核.json",summary)
    print(summary["images"],{k:v for k,v in summary["metrics"].items() if k.endswith("_DE") or k.endswith("_hidden_RR")})


if __name__=="__main__":main()
