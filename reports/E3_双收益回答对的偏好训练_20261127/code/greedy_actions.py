"""Decompose the measured R3–R2 difference by equality of the two emitted action sequences."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def summary(records):
    keys=("mR","DE","mrr")
    result={}
    for k in keys:
        rows=[r for r in records if k in r]
        if rows:
            result[k]=dict(observations=len(rows),mean_delta=float(np.mean([r[k] for r in rows])))
    return result


def main():
    fixed=read(ROOT/"validation/训练优选原文.json")["fixed_tokens"]
    scores={(d["seed"],d["arm"],d["image"]):d for d in read(ROOT/"evaluation/results/逐图指标.json") if d["arm"] in ("r2","r3")}
    data=[]
    counts={}
    for seed in "ab":
        names=sorted(n for (s,a,n) in scores if seed==s and a=="r3")
        assert len(names)==210
        counter=Counter()
        for name in names:
            logs={}
            for arm in ("r2","r3"):
                path=ROOT/"evaluation/records"/f"{seed}_{arm}"/(name+".pt")
                log=torch.load(path,map_location="cpu",weights_only=True)
                logs[arm]=log
                actions=log["actions"]
                if len(actions)>=2:
                    counter[arm+"_second_round"]+=1
                    if actions[1]==fixed:
                        counter[arm+"_fixed_second_action"]+=1
            r2,r3=[scores[seed,arm,name] for arm in ("r2","r3")]
            same=logs["r2"]["actions"]==logs["r3"]["actions"]
            same_vec=torch.equal(logs["r2"]["v"],logs["r3"]["v"])
            row=dict(seed=seed,image=name,same=same,same_visual=same_vec,mR=r3["official"]["mR"]-r2["official"]["mR"])
            if "first106" in r2:
                row.update({k:r3["first106"][k]-r2["first106"][k] for k in ("DE","mrr")})
            data.append(row)
        counts[seed]=dict(counter)
    output=dict(by_seed=counts,all=summary(data),same_actions=summary([r for r in data if r["same"]]),
                different_actions=summary([r for r in data if not r["same"]]),
                same_actions_and_identical_visual=sum(r["same"] and r["same_visual"] for r in data))
    (ROOT/"results/正式生成动作核查.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(output)


if __name__=="__main__":
    main()
