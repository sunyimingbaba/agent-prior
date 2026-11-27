"""Verify every logged group with exact per-caption thresholds and independent rank allocation."""
from fractions import Fraction
from pathlib import Path
import json
import hashlib
import math

import numpy as np

STAGE = Path(__file__).resolve().parents[1]
POOL = STAGE.parent / "E3_回填后双收益训练图筛选_20261127"

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def recompute(row):
    base=row["retrieval"]
    winners=[i for i,s in enumerate(base) if s==max(base)]
    correct=[sum(v>0 for v in ab) for ab in row["hard"]]
    weakest=list(map(min,row["hard"]))
    rr=[[Fraction(v) for v in cols] for cols in row["other_caption_rr"]]
    assert len({len(c) for c in rr})==1
    ok=[]
    n=len(winners)
    for j in range(8):
        if rr[j] and correct[j]*n>sum(correct[k] for k in winners) and weakest[j]>max(weakest[k] for k in winners):
            if all(rr[j][m]*n>=sum(rr[k][m] for k in winners) for m in range(len(rr[j]))):
                ok.append(j)
    result=base[:]
    qualified=ok+winners if ok else []
    sorted_base=sorted(base[i] for i in qualified)
    idx=0
    for value in sorted({weakest[i] for i in qualified}):
        same=[i for i in qualified if weakest[i]==value]
        local=float(np.mean(sorted_base[idx:idx+len(same)]))
        for i in same:
            result[i]=local
        idx+=len(same)
    return len(ok),result

def main():
    lock=STAGE/"training/strict_lock.json"
    res={}
    keys=("step","image","row","seed","proxy_rank","native_margin","format","retrieval","hard","other_caption_rr","trajectories","continuations","r2","adv2")
    roster=read(POOL/"scoring/locks/training.json")["rows"]
    for seed in "ab":
        folder=STAGE/"models"/(seed+"_strict")
        past=POOL/"models"/(seed+"_r2")
        previous=read(past/"steps.json")
        records=read(folder/"steps.json")
        proof=read(folder/"complete.json")
        assert proof["lock"]==sha(lock)
        assert proof["scores"]==sha(folder/"steps.json") and sha(past/"steps.json")==read(lock)["baseline_history"][seed]
        assert len(roster)==len(previous)==len(records)==24
        distinct=[]; top_changes=[]; candidates=0; challengers=0; exact=[]
        for j,(row,old) in enumerate(zip(records,previous)):
            assert row["seed"]=={"a":231000000,"b":241000000}[seed]+j*1000 and row["image"]==roster[j]["image"]
            assert len(row["trajectories"])==8
            if not distinct:
                assert all(row[k]==old[k] for k in keys)
                exact.append(j+1)
            n,result=recompute(row)
            assert len(set(row["format"]))==1
            orig=np.array(row["retrieval"])
            fmt=np.array(row["format"])
            np.testing.assert_array_equal(row["r2"],orig+fmt)
            np.testing.assert_array_equal(row["r3"],result+fmt)
            np.testing.assert_array_equal(row["bonus"],result-orig)
            assert np.isclose(sum(result),sum(orig),atol=1e-12,rtol=0)
            r3=np.array(row["r3"])
            adv=(r3-r3.mean())/(r3.std(ddof=1)+1e-8)
            np.testing.assert_array_equal(adv,row["advantage"])
            if np.any(row["bonus"]):
                assert n>0
                distinct.append(j+1)
                candidates+=int(np.count_nonzero(row["bonus"]))
                challengers+=n
                if not np.array_equal(orig==orig.max(),r3==r3.max()):
                    top_changes.append(j+1)
            elif not distinct:
                assert old["norm"]==row["norm"]
        identity=proof["final"]==read(past/"complete.json")["final"]
        if not distinct:
            assert identity
        res[seed]=dict(images=len(records),sampled_candidates=8*len(records),
            groups_with_changed_scores=distinct,groups_with_changed_top=top_changes,
            candidates_with_changed_scores=candidates,qualified_lower_candidates=challengers,
            initial_consecutive_groups_exact_token_logp_and_readout=exact,
            nonzero_gradient_updates=proof["nonzero"],final_adapter_identical_to_r2=identity)
    (STAGE/"training/audit.json").write_text(json.dumps(res,ensure_ascii=False,indent=2)+"\n")
    print(res,flush=True)

if __name__=="__main__":
    main()
