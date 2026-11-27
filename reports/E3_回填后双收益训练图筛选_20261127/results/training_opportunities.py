"""Measure opportunities and reward signals on the answers actually sampled during updates."""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/"E3_R2逐候选分值与困难细节加分_20261127"


def analyze(record):
    ab = (np.array(record["hard"])>0).sum(axis=1)
    rr = [sum(map(Fraction.from_float,values)) for values in record["other_caption_rr"]]
    orig = np.array(record["r2"])
    r3 = np.array(record["r3"])
    top = np.flatnonzero(orig==orig.max()).tolist()
    best = np.flatnonzero(r3==r3.max()).tolist()
    off = [i for i in range(8) if ab[i]*len(top)>sum(ab[t] for t in top) and
           rr[i]*len(top)>=sum(rr[t] for t in top)] if record["other_caption_rr"][0] else []
    change = np.asarray(record["adv3"])-np.asarray(record["adv2"])
    return dict(image=record["image"],step=record["step"],opportunity_slots=off,has=bool(off),
                mean_adv_delta=float(change[off].mean()) if off else None,
                captured=bool(set(best)&set(off)),weight=len(set(best)&set(off))/len(best),
                beyond_best=ab[best].mean()>ab[top].mean() and
                sum(rr[i] for i in best)*len(top)>=sum(rr[i] for i in top)*len(best))


def main():
    total = {}
    for tag,base in [("previous",PREVIOUS),("current",ROOT)]:
        out = {}
        for arm in ("r2","r3"):
            records=[]
            by_seed={}
            for seed in "ab":
                history=json.loads((base/"models"/f"{seed}_{arm}"/"steps.json").read_bytes())
                batch=[analyze(row) for row in history]
                assert len(batch)==24
                by_seed[seed]=sum(r["has"] for r in batch)
                records+=batch
            avail=[r for r in records if r["has"]]
            out[arm]=dict(total_groups=len(records),has_by_seed=by_seed,opportunities=len(avail),
                          groups_with_an_opportunity_in_r3_top=sum(r["captured"] for r in records),
                          groups_with_average_over_r3_top_joint=int(sum(r["beyond_best"] for r in records)),
                          groups_mean_opportunity_advantage_up=sum(r["mean_adv_delta"]>1e-10 for r in avail),
                          groups_mean_opportunity_advantage_down=sum(r["mean_adv_delta"]< -1e-10 for r in avail),
                          groups_mean_opportunity_advantage_unchanged=sum(abs(r["mean_adv_delta"])<=1e-10 for r in avail))
        total[tag]=out
    assert total["previous"]["r3"]["opportunities"]==3
    (ROOT/"results/实际训练组的兼顾候选.json").write_text(json.dumps(total,indent=2,ensure_ascii=False)+"\n")
    print(total)


if __name__=="__main__":
    main()
