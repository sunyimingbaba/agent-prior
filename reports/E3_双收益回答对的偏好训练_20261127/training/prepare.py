"""Verify the sealed pair manifest and source trajectories, then freeze training inputs."""
import hashlib
import json
from pathlib import Path
import sys

stage=Path(__file__).resolve().parents[1]
pool=stage.parent/"aftertool_joint_pool_20261127"
source=stage.parent/"second_answer_5090_20261124"


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    with p.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def seal(path,obj):
    if path.exists():
        assert read(path)==obj,path
        return
    tmp=path.with_suffix(".part")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(path)


def prepare():
    input_path=stage/"inputs/train_pairs.json"
    assert sha(input_path)==(stage/"inputs/train_pairs.sha256").read_text().strip().split()[0]
    manifest=read(input_path)["rows"]
    assert 2<=len(manifest)<=24 and len({r["image"] for r in manifest})==len(manifest)
    roster={r["image"]:r for r in read(pool/"sampling/inputs/roster.json")}
    sample_proof=read(pool/"sampling/a_complete.json")
    score_proof=read(pool/"scoring/locks/a_selection.json")
    rows=[]
    for r in manifest:
        n=r["image"]
        base=roster[n]
        assert r["source_seed"]=="a" and (r["row"],r["rgb_sha256"],r["pair"])==(base["row"],base["rgb_sha256"],base["pair"])
        assert sha(pool/"sampling/raw/a"/(n+".pt"))==sample_proof["files"][n]==r["raw_sha256"]
        assert sha(pool/"scores/a"/(n+".json"))==score_proof["groups"][n]==r["score_sha256"]
        assert sample_proof["eligibility"][n]
        assert r["orig_query"]==base["orig_query"] and r["feedback"]==base["feedback"]
        assert r["kind"] in ("joint","anchor") and (r["r2_winner"]!=r["r3_winner"])==(r["kind"]=="joint")
        w,l=r["r2_winner"],r["r2_loser"]
        assert {w,l}=={r["r3_winner"],r["r3_loser"]} and w!=l
        assert all(isinstance(i,int) and 0<=i<8 for i in (w,l))
        slots=sorted((w,l))
        rows.append(dict(image=n,row=r["row"],rgb_sha256=r["rgb_sha256"],pair=r["pair"],kind=r["kind"],
                         raw_sha256=r["raw_sha256"],score_sha256=r["score_sha256"],orig_query=base["orig_query"],
                         feedback=base["feedback"],slots=slots,preferences={a:slots.index(r[a+"_winner"]) for a in ("r2","r3")}))
    assert sum(r["preferences"]["r2"]!=r["preferences"]["r3"] for r in rows)*2==len(rows)
    data=stage/"training/data.json"
    seal(data,rows)
    files=("training/dpo.py","training/prepare.py","training/run.sh","training/训练口径.md","inputs/train_pairs.json","training/data.json")
    lock={"stage":{f:sha(stage/f) for f in files},"historical":{},"pairs":len(rows),"sampling":sha(pool/"sampling/a_complete.json")}
    expected=read(stage.parent/"graded_r2_detail_20261127/inputs/train_lock.json")
    used=("code/policy.py","code/shared.py","initial.json","R0/adapter/adapter_config.json","R0/adapter/adapter_model.safetensors")
    for seed,folder in (("a",source),("b",source/"repeat")):
        lock["historical"][seed]={n:sha(folder/n) for n in used}
        assert all(expected[seed][n]==h for n,h in lock["historical"][seed].items())
    seal(stage/"training/lock.json",lock)
    print("locked",len(rows),"pairs",flush=True)


if __name__=="__main__":
    prepare()
