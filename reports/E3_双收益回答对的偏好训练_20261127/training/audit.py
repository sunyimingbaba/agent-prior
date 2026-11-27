"""Independently recompute DPO loss and validate the exact tokens and both preference directions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

stage=Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def audit():
    lock=stage/"training/lock.json"
    rows=read(stage/"training/data.json")
    result={}
    for seed in "ab":
        rf=stage/"training"/("reference_"+seed+".json")
        baseline=read(rf)
        assert len(baseline["pairs"])==len(rows) and baseline["lock"]==sha(lock)
        info={}
        for arm in ("r2","r3"):
            d=stage/"models"/f"{seed}_{arm}"
            history=read(d/"steps.json")
            proof=read(d/"complete.json")
            assert len(history)==len(rows)==proof["steps"] and proof["nonzero"]==sum(x["norm"]>0 for x in history)
            assert proof["reference"]==sha(rf) and proof["initial"]==baseline["initial"] and proof["lock"]==baseline["lock"]
            assert proof["final"]==sha(d/"adapter/adapter_model.safetensors") and proof["scores"]==sha(d/"steps.json")
            assert proof["train"]==sha(stage/"training/data.json")
            for i,(row,ref,actual) in enumerate(zip(rows,baseline["pairs"],history)):
                chosen=row["preferences"][arm]
                assert (i+1,actual["image"],actual["slots"],chosen)==(actual["step"],row["image"],row["slots"],actual["chosen_position"])
                assert actual["slots"]==ref["slots"] and [r["tokens"] for r in actual["responses"]]==[r["tokens"] for r in ref["responses"]]
                assert actual["chosen_slot"]==row["slots"][chosen] and actual["rejected_slot"]==row["slots"][1-chosen]
                for z in [*actual["responses"],*ref["responses"]]:
                    assert len(z["per_token"])==len(z["tokens"]) and all(np.isfinite(z["per_token"])) and max(z["per_token"])<=0
                    assert math.isclose(z["logp"],math.fsum(z["per_token"]),rel_tol=1e-6,abs_tol=1e-4)
                diff=actual["responses"][chosen]["logp"]-actual["responses"][1-chosen]["logp"]
                frozen=ref["responses"][chosen]["logp"]-ref["responses"][1-chosen]["logp"]
                logit=0.1*(diff-frozen)
                assert actual["policy_gap"]==diff and actual["reference_gap"]==frozen and actual["logit"]==logit
                assert math.isclose(actual["loss"],np.logaddexp(0,-logit),rel_tol=1e-12,abs_tol=1e-12)
                assert math.isclose(actual["gradient_scale"],0.1*math.exp(-np.logaddexp(0,logit)),rel_tol=1e-12,abs_tol=1e-12)
            assert history[0]["responses"]==baseline["pairs"][0]["responses"] and history[0]["loss"]==math.log(2)
            info[arm]=history
            result[f"{seed}_{arm}"]=dict(groups=len(history),nonzero=proof["nonzero"],initial=proof["initial"],final=proof["final"],
                                          reference_sha256=sha(rf),steps_sha256=proof["scores"])
        assert all(a["image"]==b["image"] and a["slots"]==b["slots"] for a,b in zip(info["r2"],info["r3"]))
    target=stage/"training/audit.json"
    tmp=target.with_suffix(".part")
    tmp.write_text(json.dumps(result,indent=2)+"\n")
    tmp.replace(target)


if __name__=="__main__":
    audit()
