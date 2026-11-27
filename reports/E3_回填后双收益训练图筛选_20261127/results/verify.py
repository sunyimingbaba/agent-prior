"""Independently reconstruct all per-group winner and opportunity labels."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(group):
    items = group["items"]
    margins = np.array([x["margin"] for x in items])
    proxy = np.sort([1.0/(1+x["rank"]) for x in items])
    ab = np.array([x["hard_margins"] for x in items])
    weak = np.maximum(ab.min(axis=1),0)
    fine = (ab>0).sum(axis=1)
    costs = [sum(map(Fraction.from_float,x["other_caption_rr"])) for x in items]
    r2 = np.array([proxy[sum(margins<m):sum(margins<=m)].mean() for m in margins])
    middle = np.array([sum(weak<w)+(sum(weak==w)-1)/2 for w in weak])
    bonus = np.zeros(8) if np.ptp(middle)==0 else (np.ptp(r2)/7)*((middle-middle.min())/np.ptp(middle))
    assert np.array_equal(r2,group["r2"]) and np.allclose(bonus,group["bonus"],rtol=1e-14,atol=0)
    top = np.flatnonzero(r2==max(r2)).tolist()
    final = np.asarray(group["r3"])
    assert np.array_equal(r2+np.asarray(group["bonus"]),final)
    new = np.flatnonzero(final==max(final)).tolist()
    assert top==group["r2_top"] and new==group["r3_top"]
    target = sum(costs[i] for i in top)
    slots = [i for i in range(8) if fine[i]*len(top)>sum(fine[t] for t in top) and
             costs[i]*len(top)>=target] if items[0]["other_caption_rr"] else []
    assert slots==group["opportunity_slots"]
    assert group["achieved"]==bool(set(slots)&set(new))
    return bool(slots), len(set(slots)&set(new))/len(new)


def main():
    rows = read(ROOT/"sampling/inputs/roster.json")
    outcomes = {}
    mechanisms = {}
    for seed in "ab":
        manifest=read(ROOT/"sampling"/(seed+"_complete.json"))
        per_image={}
        for row in rows:
            name=row["image"]
            record=read(ROOT/"scores"/seed/(name+".json"))
            assert record["source"]==manifest["files"][name]
            assert record["eligible"]==manifest["eligibility"][name]
            assert record["rgb"]==row["rgb_sha256"]
            if record["eligible"]:
                per_image[name]=verify(record)
                if seed == "b" and record["opportunity_slots"]:
                    slots = set(record["opportunity_slots"])
                    top = set(record["r2_top"])
                    final = set(record["r3_top"])
                    mechanisms[name] = dict(all_r2_tied=np.ptp(record["r2"])==0,
                                            all_bonuses_tied=np.ptp(record["bonus"])==0,
                                            opportunity_in_r2_top=bool(slots & top),
                                            winner_set_changed=final!=top,
                                            all_new_winners_good=final<=slots,
                                            changed_and_all_new_winners_good=final!=top and final<=slots,
                                            changed_and_some_new_winners_good=final!=top and bool(final&slots))
        outcomes[seed]=per_image
    names = sorted(set(outcomes["a"]) & set(outcomes["b"]))
    counts = {str((a,b)):sum(outcomes["a"][n][0]==a and outcomes["b"][n][0]==b for n in names)
              for a in (False,True) for b in (False,True)}
    report = dict(groups_checked=sum(len(x) for x in outcomes.values()),common=len(names),want_A_and_B=counts,
                  b_groups_with_opportunity=sum(outcomes["b"][n][0] for n in names),
                  b_recovered_by_r3_tie_weighted=sum(outcomes["b"][n][1] for n in names),
                  b_recovered_by_r3_any=sum(outcomes["b"][n][1]>0 for n in names))
    useful = [mechanisms[name] for name in names if name in mechanisms]
    report["b_opportunity_groups_breakdown"] = {key:sum(int(row[key]) for row in useful) for key in useful[0]} if useful else {}
    (ROOT/"results/独立逐组核对.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    main()
