"""Recompute every recorded candidate score and verify matched arms before formal evaluation."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

stage = Path(__file__).resolve().parents[1]
previous = stage.parent/"graded_r2_detail_20261127"
sys.path.insert(0,str(previous/"code"))
from detail_reward import graded_bonus


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    roster_file=stage/"scoring/locks/training.json"
    roster=read(roster_file)["rows"]
    lock_path=stage/"training/train_lock.json"
    lock=read(lock_path)
    report={}
    for seed,base_rng in (("a",231000000),("b",241000000)):
        histories={}
        for arm in ("r2","r3"):
            path=stage/"models"/f"{seed}_{arm}"
            proof=read(path/"complete.json")
            history=read(path/"steps.json")
            assert proof["steps"]==len(history)==len(roster)==lock["n"]
            assert all(proof[k]==sha(p) for k,p in (
                ("lock",lock_path),("scores",path/"steps.json"),("train",roster_file),
                ("final",path/"adapter/adapter_model.safetensors")))
            assert proof["initial"]==lock["historical"][seed]["R0/adapter/adapter_model.safetensors"]
            assert proof["nonzero"]==sum(row["norm"]>0 for row in history)
            for i,(row,case) in enumerate(zip(history,roster)):
                assert (row["step"],row["seed"],row["image"],row["row"])==(i+1,base_rng+i*1000,case["image"],case["row"])
                margins=np.array(row["native_margin"])
                ranks=np.array(row["proxy_rank"])
                assert margins.shape==ranks.shape==(8,) and np.isfinite(margins).all() and np.all(ranks>=0)
                ordered=np.sort(1/(ranks+1))
                counted=np.array([ordered[np.count_nonzero(margins<m):np.count_nonzero(margins<=m)].mean() for m in margins])
                assert np.array_equal(counted,row["retrieval"])
                fmt=np.array(row["format"])
                assert fmt.shape==(8,) and np.ptp(fmt)==0
                bonus=graded_bonus(counted,row["hard"])
                assert np.array_equal(bonus,row["bonus"])
                base=fmt+counted
                rewards=base+bonus
                assert np.array_equal(base,row["r2"]) and np.array_equal(rewards,row["r3"])
                source=base if arm=="r2" else rewards
                assert np.array_equal((source-source.mean())/(source.std(ddof=1)+1e-8),row["advantage"])
                assert len(row["trajectories"])==len(row["continuations"])==8
                assert all(len(t["ids"])==len(t["sampled_logp"])>0 for t in row["trajectories"])
            histories[arm]=history
            report[f"{seed}_{arm}"]=dict(nonzero=proof["nonzero"],steps=len(history),adapter=proof["final"],
                history=proof["scores"],max_replay_logp_error=max(r["replay_max_abs_logp_difference"] for r in history))
        a,b=histories["r2"][0],histories["r3"][0]
        fields=("trajectories","continuations","other_caption_rr","proxy_rank","retrieval","native_margin","format","r2","r3","hard","bonus")
        assert all(a[field]==b[field] for field in fields)
    out=stage/"training/audit.json"
    tmp=out.with_suffix(".part")
    tmp.write_text(json.dumps(report,indent=2)+"\n")
    tmp.replace(out)


if __name__=="__main__":
    main()
