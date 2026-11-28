"""Lock the third-draw PRE-sampling roster and all four training/assessment programs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT.parent
REPORTS=STAGE.parent
POOL=REPORTS/"E3_回填后双收益训练图筛选_20261127"
THIRD=STAGE/"training_signal/third"
CAL=REPORTS/"E3_困难奖励跨图跨措辞校准_20261127"
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(path):return json.loads(path.read_bytes())
def save(path,val):path.write_text(json.dumps(val,indent=2,ensure_ascii=False)+"\n")

def main():
    lock=read(THIRD/"inputs/roster_lock.json")
    assert sha(THIRD/"inputs/roster.json")==lock["roster"] and sha(THIRD/"inputs/groups.json")==lock["groups"]
    rows=read(THIRD/"inputs/roster.json")
    assert rows==sorted(rows,key=lambda r:r["rgb_sha256"])
    assert len(rows)==len({r["rgb_sha256"] for r in rows})==24
    consensus=read(THIRD/"inputs/groups.json")
    assert set(r["image"] for r in rows)==set(r["image"] for r in consensus)
    assert len({(r["cohort"],r["pair"]) for r in consensus})==12
    roster=ROOT/"inputs/roster.json"
    save(roster,dict(rows=rows,groups=consensus))
    previous=POOL/"training/train_lock.json"
    c={"roster":sha(roster),"third_roster_lock":sha(THIRD/"inputs/roster_lock.json"),
       "pool_lock":sha(previous),"redistribution":sha(CAL/"code/reward.py"),"n":24,"starts":{},"contexts":{}}
    for s in "ab":
        record=read(POOL/"sampling"/(s+"_complete.json"))
        c["starts"][s]=record["adapter"]
        assert c["starts"][s]==read(previous)["historical"][s]["R0/adapter/adapter_model.safetensors"]
        assert all(record["eligibility"][r["image"]] for r in rows)
        c["contexts"][s]={r["image"]:record["files"][r["image"]] for r in rows}
    code=sorted((ROOT/"training").glob("*.py"))+sorted((ROOT/"evaluation/code").glob("*.py"))
    code += [ROOT/"事前口径.md", ROOT/"evaluation/验收口径.md",ROOT/"training/queue.sh",roster]
    c["files"]={str(f.relative_to(ROOT)):sha(f) for f in code}
    save(ROOT/"inputs/train_lock.json",c)
if __name__=="__main__":main()
