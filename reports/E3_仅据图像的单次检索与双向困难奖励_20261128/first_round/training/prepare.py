"""Seal all training inputs before the first on-policy first-round draw."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MATCHED=ROOT.parent/"matched"
REPORTS=ROOT.parents[1]
POOL=REPORTS/"E3_回填后双收益训练图筛选_20261127"
CAL=REPORTS/"E3_困难奖励跨图跨措辞校准_20261127"
def sha(p):
    with p.open("rb") as s:return hashlib.file_digest(s,"sha256").hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n")
if __name__=="__main__":
    older=MATCHED/"inputs/train_lock.json";prior=read(older)
    roster=ROOT/"inputs/roster.json"
    roster.write_bytes((MATCHED/"inputs/roster.json").read_bytes())
    assert sha(roster)==prior["roster"]
    assert sha(POOL/"training/train_lock.json")==prior["pool_lock"]
    assert sha(CAL/"code/reward.py")==prior["redistribution"]
    assert sha(POOL/"scoring/locks/training.json")==prior["historical_roster"]
    files=sorted((ROOT/"training").glob("*.py"))+[ROOT/"training/queue.sh",ROOT/"训练事前口径.md",roster]
    previous_trace=REPORTS/"E3_UCM候选来源与单次检索推进_20261125/code/full_traces.py"
    save(ROOT/"inputs/train_lock.json",{**{k:prior[k] for k in ("roster","starts","pool_lock","redistribution","seed_schedule","query")},
         "sampler":sha(previous_trace),"old_input_lock":sha(older),"feedback":sha(MATCHED/"inputs/feedback_verified.json"),
         "reference_contexts":{s:sha(MATCHED/"contexts"/s/"complete.json") for s in "ab"},
         "files":{str(f.relative_to(ROOT)):sha(f) for f in files}})
