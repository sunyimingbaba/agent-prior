"""Lock the first published 24-image selection, image-only programs and unchanged scoring assets."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parents[1]
POOL=REPORTS/"E3_回填后双收益训练图筛选_20261127"
CAL=REPORTS/"E3_困难奖励跨图跨措辞校准_20261127"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,z):p.write_text(json.dumps(z,indent=2,ensure_ascii=False)+"\n")
if __name__=="__main__":
    source=POOL/"scoring/locks/training.json";prior=POOL/"training/train_lock.json"
    assert sha(source)==read(prior)["training_inputs"]["scoring/locks/training.json"]
    rows=[{k:r[k] for k in ("image","row","rgb_sha256","split")} for r in read(source)["rows"]]
    assert len(rows)==len(set(r["rgb_sha256"] for r in rows))==24 and rows==sorted(rows,key=lambda r:r["rgb_sha256"])
    roster=ROOT/"inputs/roster.json";save(roster,dict(rows=rows))
    control=ROOT.parent/"image_only/inputs/manifest.json"
    query=read(control)["query"]
    assert query=="Describe the visible features of this remote sensing image."
    scripts=sorted((ROOT/"training").glob("*.py"))+[ROOT/"training/queue.sh",ROOT/"训练事前口径.md",roster]
    save(ROOT/"inputs/train_lock.json",dict(query=query,roster=sha(roster),seed_schedule={"a":231000000,"b":241000000},
         starts={s:read(prior)["historical"][s]["R0/adapter/adapter_model.safetensors"] for s in "ab"},
         pool_lock=sha(prior),historical_roster=sha(source),redistribution=sha(CAL/"code/reward.py"),
         prior_feedback=sha(POOL/"sampling/feedback_verified.json"),image_only_manifest=sha(control),
         files={str(p.relative_to(ROOT)):sha(p) for p in scripts}))
