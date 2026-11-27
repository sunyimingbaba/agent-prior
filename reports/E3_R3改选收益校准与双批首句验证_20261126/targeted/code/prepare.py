"""Export the exact two source text banks and validate all training image IDs."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT / "code"))
from data import PREV,read,previous_banks,sha
from training_opportunities import winners
import training_opportunities
from target_reward import TargetReward


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")


def main():
    rows = sorted((r for r in read(PARENT/"cold/inputs/real_roster.json") if r["status"]=="selected"),key=lambda r:r["rgb_sha256"])
    assert len({r["rgb_sha256"] for r in rows})==len(rows)==13
    prev=read(PREV/"锁定评分参数.json")["model"]
    w = np.array(prev["coef"]) / np.array(prev["scale_x"])[:,None]
    banks = previous_banks()
    pack = {"coefficients":w}
    for i, bank in enumerate(banks):
        pack["native"+str(i)] = bank["native"]
        pack["odd"+str(i)] = ~bank["even"]
    mapping={}
    for r in rows:
        source=int(r["tag"] in ("saved","extra"))
        name=r["image"];b=banks[source];cols=b["maps"][name]
        gg=np.array([cols[k] for k in "ABC"])
        pack[name]=b["hard"][gg.ravel()]
        mapping[name]={"bank":source,"native":[int(i) for i in cols["native"]],"ABC":gg.tolist()}
    np.savez_compressed(ROOT/"inputs/text.npz",**pack)
    dump(ROOT/"inputs/index.json",mapping)
    dump(ROOT/"inputs/pilot/train.json",{s:rows for s in "ab"})
    scorer=TargetReward("a"); references={}
    for r in rows:
        n=r["image"]; references[n]={}
        b=banks[mapping[n]["bank"]]
        for s in "ab":
            orig=PARENT/"cold/records/real"/s
            pf=orig/"features"/(n+".pt")
            mf={k:v for f in (orig/"features").glob("complete*.json") for k,v in read(f)["files"].items()}
            assert sha(pf)==mf[n]
            v=torch.load(pf,weights_only=True)["visual"].double().numpy()
            gold=[];ref_fn=training_opportunities.old_choices
            def observe(pred,p):
                result=ref_fn(pred,p)
                gold.append((p["ref"],result))
                return result
            with patch.object(training_opportunities,"old_choices",observe):
                winners(r,b,v,w)
            assert len(gold)==4
            expected=[]
            for k in range(2):
                assert gold[2*k][0]==gold[2*k+1][0]
                z=dict(ordinary=gold[2*k][0],old=gold[2*k][1],old3=gold[2*k+1][1])
                assert z==scorer.run(n,v[8*k:8*k+8])
                expected.append(z)
            references[n][s]=expected
    dump(ROOT/"inputs/expected.json",references)
    tracked=[p for d in ("code","inputs") for p in (ROOT/d).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="decision.json"]+[ROOT/"训练口径.md"]
    receipt=dict(starts=read(PARENT/"inputs/pilot/decision.json")["starts"],train_arms=["ordinary","old","old3"],
        groups_checked=sum(2*len(z) for z in references.values()),lock={str(p.relative_to(ROOT)):sha(p) for p in tracked})
    dump(ROOT/"inputs/pilot/decision.json",receipt)


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
