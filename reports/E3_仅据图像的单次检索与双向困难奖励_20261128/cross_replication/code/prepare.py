"""Strip query captions out of the prospective generation roster and reproduce the UCM donor rule."""
from collections import Counter
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT.parent;REPORTS=STAGE.parent
OLD=REPORTS/"E3_视觉细节复核与RFPO_20261127/cross448"
FORM=REPORTS/"E3-2_UCM与RSITMD独立复验_20261124"
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    known=read(OLD/"inputs/manifest.json");rs=FORM/"inputs/rsitmd.json";tools=FORM/"records/tools.json"
    assert sha(rs)==known["roster"] and sha(tools)==known["tool_cache"]
    allrows=read(rs);lookup=read(tools)
    rows=[dict(image=r["image"],row=r["row"],rgb=r["rgb_sha256"],feedback=lookup[r["rgb_sha256"]][0]) for r in allrows]
    assert len(rows)==len({r["rgb"] for r in rows})==448
    for roster in (STAGE/"first_round/inputs/roster.json",STAGE/"matched/inputs/roster.json"):
        assert {r["rgb_sha256"] for r in read(roster)["rows"]}.isdisjoint(r["rgb"] for r in rows)
    def top(row):return row["feedback"].split(" (",1)[0]
    sorted_rows=sorted(rows,key=lambda r:(top(r),r["rgb"]));distribution=Counter(map(top,sorted_rows))
    shift=max(distribution.values());assert shift<len(rows)/2
    assignment={r["rgb"]:sorted_rows[(i+shift)%len(rows)]["rgb"] for i,r in enumerate(sorted_rows)}
    by={r["rgb"]:r for r in rows};assert Counter(assignment.values())==Counter(assignment.keys())
    assert all(top(by[a])!=top(by[b]) for a,b in assignment.items())
    save(ROOT/"inputs/roster.json",rows);save(ROOT/"inputs/donors.json",assignment)
    ev=read(STAGE/"scene_evidence/inputs/manifest.json")
    oldfiles={"old448":OLD/"inputs/manifest.json","weights_source":STAGE/"image_only/inputs/manifest.json",
              "scene_method":STAGE/"scene_evidence/inputs/manifest.json","first_train":STAGE/"first_round/inputs/train_lock.json",
              "first_eval":STAGE/"first_round/evaluation/inputs/evaluation_lock.json",
              "evidence_generate":STAGE/"scene_evidence/code/generate.py","query_check":STAGE/"image_only/code/predict.py",
              "old448_score":OLD/"code/score.py","gallery_text":FORM/"results/rsitmd_text.pt"}
    files=sorted((ROOT/"code").glob("*.py"))+[ROOT/"code/queue.sh",ROOT/"复核事前口径.md",ROOT/"inputs/roster.json",ROOT/"inputs/donors.json"]
    save(ROOT/"inputs/manifest.json",dict(query=ev["query"],label=ev["label"],images=448,captions=2240,overlap=0,
         donor_shift=shift,scene_counts=dict(distribution),old_policies=ev["policies"],
         prior_gallery=known["roster"],arrow=known["arrow"],tool_cache=known["tool_cache"],head=known["scene_head"],
         scorer_sources=known["scorer_sources"],external={k:sha(v) for k,v in oldfiles.items()},
         files={str(v.relative_to(ROOT)):sha(v) for v in files}))
