from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from reward import unit

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parent
PAIRS=REPORTS/"E3_双收益回答对的偏好训练_20261127/validation98"
OLD=PAIRS/"受控真假回答"
POOL=REPORTS/"E3_回填后双收益训练图筛选_20261127"


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def tensor(p):
    return torch.load(p,weights_only=True,map_location="cpu")


def reference():
    return read(PAIRS/"fresh98/inputs/rows.json")


def fit(rows,state_fn):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        h=unit(state_fn(r))
        assert h.shape==(2,3584)
        groups[r["attribute"]][r["pair"]].append((1-2*r["side"])*(h[0]-h[1]))
    out={}
    for key,group in groups.items():
        assert all(len(v)==2 for v in group.values())
        out[key]=unit(np.mean([np.mean(v,axis=0) for v in group.values()],axis=0))
    return out


def interval(rows):
    keys=sorted({(r["cohort"],r["pair"]) for r in rows})
    rng=np.random.default_rng(271127)
    weights=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=20000)
    index={k:i for i,k in enumerate(keys)}
    fields=list(rows[0]["metrics"])
    arr=np.zeros((len(keys),len(fields)+1))
    for r in rows:
        arr[index[r["cohort"],r["pair"]]] += [*[r["metrics"][k] for k in fields],1]
    b=weights@arr
    cis=np.quantile(b[:,:-1]/b[:,-1,None],[.025,.975],axis=0).T
    return dict(images=len(rows),pairs=len(keys),metrics={k:dict(mean=float(val),ci=ci.tolist()) for k,val,ci in
                      zip(fields,arr[:,:-1].sum(0)/len(rows),cis)})


def main():
    train=reference()
    locked=read(OLD/"generation.json")
    def get(r):
        p=OLD/"raw"/(r["image"]+".pt")
        z=tensor(p)
        assert z["rgb"]==r["rgb_sha256"] and z["texts"]==["<answer>"+t+"</answer>" for t in r["A"]]
        return z["states"].float().numpy()
    directions=fit(train,get)
    np.savez(ROOT/"inputs/directions.npz",**directions)
    full=read(POOL/"scoring/inputs/rows.json")
    eligible=read(POOL/"sampling/a_complete.json")["eligibility"]
    eb=read(POOL/"sampling/b_complete.json")["eligibility"]
    trainwords={(r["attribute"],r["side"],tuple(r["A"])) for r in train}
    oldpix={r["rgb_sha256"] for r in train}
    reserved={(r["cohort"],r["pair"]) for r in read(POOL/"scoring/locks/training.json")["rows"]}
    groups=defaultdict(list)
    for r in full:
        assert r["rgb_sha256"] not in oldpix
        if r["attribute"] in directions:
            assert (r["attribute"],r["side"],tuple(r["A"])) in trainwords
            groups[r["cohort"],r["pair"]].append(r)
    data=[r for k,rs in groups.items() if len(rs)==2 and all(eligible[x["image"]] and eb[x["image"]] for x in rs) for r in rs]
    assert len(data)==238
    out=[r for r in data if (r["cohort"],r["pair"]) not in reserved]
    save(ROOT/"inputs/test_panel.json",out)
    save(ROOT/"inputs/读出来源.json",dict(train_images=len(train),train_attributes=sorted(directions),prior_manifest=sha(OLD/"generation.json"),
         training_files={r["image"]:sha(OLD/"raw"/(r["image"]+".pt")) for r in train},target_images=len(data),main=len(out),
         main_pairs=len({(r["cohort"],r["pair"]) for r in out}),natural=sum(len(r["alt_ids"])>=2 for r in out),
         reserved_pairs=sorted(reserved),direction_sha256=sha(ROOT/"inputs/directions.npz")))
    print(len(out),len(directions))


if __name__=="__main__": main()
