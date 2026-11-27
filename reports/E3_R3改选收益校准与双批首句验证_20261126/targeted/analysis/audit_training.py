"""Verify locked training order and identify when each pair of policies first diverged."""
from pathlib import Path
import hashlib
import json
import math
import itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt=ROOT/"inputs/pilot/decision.json";lock=read(receipt)
    rows=read(ROOT/"inputs/pilot/train.json")
    assert rows["a"]==rows["b"]==sorted(rows["a"],key=lambda r:r["rgb_sha256"])
    result={}
    for seed in "ab":
        records={};counts={};prefix={}
        for arm in lock["train_arms"]:
            folder=ROOT/"models"/f"{seed}_{arm}"
            steps=read(folder/"steps.json");proof=read(folder/"complete.json")
            assert proof["lock"]==digest(receipt) and proof["initial"]==lock["starts"][seed]
            assert proof["scores"]==digest(folder/"steps.json") and proof["steps"]==len(steps)==13
            assert proof["nonzero"]==sum(r["norm"]>0 for r in steps)
            for j,(step,row) in enumerate(zip(steps,rows[seed])):
                assert step["step"]==j+1 and step["image"]==row["image"] and step["query"]==row["orig_query"]
                assert step["rng"]==(165000000 if seed=="a" else 175000000)+j*1000
                assert len(step["paths"])==8 and len(step["states"])==64
                assert math.isfinite(step["norm"]) and math.isfinite(step["max_replay_error"])
                winners=step["winners"][arm]
                score=np.zeros(8);score[winners]=1
                assert np.allclose(step["advantages"],(score-score.mean())/(score.std(ddof=1)+1e-8),atol=1e-12)
            counts[arm]={"nonzero":proof["nonzero"],"final":proof["final"],
                "max_replay_error":max(s["max_replay_error"] for s in steps),
                "mean_replay_error":float(np.mean([s["mean_replay_error"] for s in steps])),
                "different_winners":{f"{a}-{b}":[r["step"] for r in steps if r["winners"][a]!=r["winners"][b]]
                      for a,b in itertools.combinations(lock["train_arms"],2)}}
            records[arm]=steps
        for a,b in itertools.combinations(lock["train_arms"],2):
            first=None
            for x,y in zip(records[a],records[b]):
                assert all(x[k]==y[k] for k in ("paths","states","winners","unique","image","rng","query"))
                if x["advantages"]!=y["advantages"]:
                    first=x["step"]
                    break
                assert all(x[k]==y[k] for k in ("norm","max_replay_error","mean_replay_error"))
            prefix[f"{a}-{b}"]={"first_different_advantage":first,"identical_updates_before":first-1 if first else 13}
        result[seed]={"arms":counts,"deterministic_prefix":prefix}
    (ROOT/"results/训练核查.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
