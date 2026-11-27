"""Check the fixed within-tie alternative on a separate historical first-caption panel."""
from compare import OLD, REPORTS, ROOT, LIBRARIES, read, sha, scores_for_seed, summary
import gzip
import json
import numpy as np


def classify(row):
    scores = np.array(row["r2"])
    bonus = np.array(row["bonus"])
    gaps = np.diff(np.unique(scores))
    cap = gaps.min()/2 if len(gaps) else 0.
    scale = min(1.,cap/bonus.max()) if bonus.max()>0 else 1.
    safe = scores + scale*bonus
    base = scores==scores.max()
    selected=safe==safe.max()
    main=np.equal(scores+bonus,np.max(scores+bonus))
    assert np.all(base[selected]) and not np.any((scores[:,None]>scores[None,:]) & (safe[:,None]<=safe[None,:]))
    values = {"tie_only_switch":int(not np.array_equal(base,selected)),
              "main_switch":int(not np.array_equal(base,main)),
              "main_lower_enters":int(any(~base[main]))}
    for k in ("DE","mrr"):
        if k not in row["outcomes"]:
            continue
        ys=np.asarray(row["outcomes"][k])
        for n,mask in (("base",base),("main",main),("tie_only",selected)):
            values[f"{k}_{n}"]=float(ys[mask].mean())
        for n,a,b in (("main_minus_base",main,base),("tie_minus_base",selected,base),("tie_minus_main",selected,main)):
            values[f"{k}_{n}"]=float(ys[a].mean()-ys[b].mean())
    return {**row,"numbers":values,"tiebreak_winners":np.flatnonzero(selected).tolist()}


def main():
    new_source=REPORTS/"E3_R3首位配文保护与首句复验_20261126"
    roster=read(new_source/"inputs/real_roster.json")
    assert sha(new_source/"inputs/real_roster.json")==read(new_source/"inputs/real_lock.json")["roster"]
    used=set()
    for path in (OLD/"inputs/real_roster.json",REPORTS/"E3-2_UCM同起点三奖励短训_20261120/inputs/train.json",REPORTS/"E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json"):
        used.update(r["rgb_sha256"] for r in read(path))
    assert len(roster)==98 and all(r["rgb_sha256"] not in used for r in roster)
    data=[dict(r,cohort="history98") for r in roster]
    libs={"history98":LIBRARIES["v122"].parent/"reward_confirm"}
    obs=[classify(x) for s in "ab" for x in scores_for_seed(data,s,new_source,libs,True)]
    (ROOT/"candidates/分差核对98图.json").write_text(json.dumps(summary(obs),indent=2)+"\n")
    with gzip.open(ROOT/"candidates/分差核对98逐组.jsonl.gz","wt") as f:
        for r in obs:f.write(json.dumps(r)+"\n")


if __name__=="__main__":
    main()
