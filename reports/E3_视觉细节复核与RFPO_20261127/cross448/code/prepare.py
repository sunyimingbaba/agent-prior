"""Freeze the four historical DPO policies, RSITMD inputs, and paired scoring before inference."""
from pathlib import Path
import hashlib
import json

import torch

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parents[1]
FORM=REPORTS/"E3-2_UCM与RSITMD独立复验_20261124"
DPO=REPORTS/"E3_双收益回答对的偏好训练_20261127"
X=REPORTS/"E3_UCM候选来源与单次检索推进_20261125"
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")

def main():
    rfile=FORM/"inputs/rsitmd.json"
    rows=read(rfile)
    old=read(FORM/"inputs/audit.json")
    tools=read(FORM/"records/tools.json")
    assert sha(rfile)==old["regions"]["rsitmd"]["manifest"]
    assert sha(FORM/"records/tools.json")==read(FORM/"records/environment.json")["tools"]
    train=read(DPO/"inputs/train_pairs.json")["rows"]
    pixels={r["rgb_sha256"] for r in rows}
    assert len(pixels)==len(rows)==448 and all(r["orig_query"]==r["captions"][0] and len(r["captions"])==5 and r["rgb_sha256"] in tools for r in rows)
    trainpix={r["rgb_sha256"] for r in train}
    assert len(trainpix)==24 and pixels.isdisjoint(trainpix)
    text=FORM/"results/rsitmd_text.pt"
    bank=torch.load(text,map_location="cpu",weights_only=True)
    tokens=bank["native_tokens"]
    assert torch.unique(tokens,dim=0).shape==tokens.shape
    ix=bank["caption_order"].view(448,5)
    eligible=ix!=ix[:,:1]
    assert int(eligible.any(1).sum())==446 and int(eligible.sum())==1739
    controls={}
    for seed in "ab":
        for arm in ("r2","r3"):
            tag=f"{seed}_{arm}"
            cp=DPO/"models"/tag/"complete.json"
            z=read(cp)
            assert z["steps"]==24 and z["lock"]==sha(DPO/"training/lock.json") and z["scores"]==sha(cp.parent/"steps.json")
            assert z["train"]==sha(DPO/"training/data.json")
            assert z["reference"]==sha(DPO/"training"/f"reference_{seed}.json")
            controls[tag]=dict(adapter=z["final"],train_proof=sha(cp),initial=z["initial"],scores=z["scores"])
    assert all(controls[s+"_r2"]["initial"]==controls[s+"_r3"]["initial"] for s in "ab")
    files=["口径.md","code/prepare.py","code/predict.py","code/run.sh","code/score.py"]
    out=dict(policies=controls,roster=sha(rfile),gallery_text=sha(text),arrow=old["files"]["rsitmd_test"],
        tool_cache=sha(FORM/"records/tools.json"),scene_head=read(FORM/"records/environment.json")["scene_head"],
        train_pairs=sha(DPO/"inputs/train_pairs.json"),historical_scores=sha(FORM/"results/paired_results.json"),
        train_photos=len(trainpix),heldout_photos=len(pixels),exact_pixel_overlap=0,eligible_photos=446,eligible_captions=1739,
        scorer_sources={n:sha(X/"code"/n) for n in ("single_scores.py","transfer_scores.py")},
        files={x:sha(ROOT/x) for x in files})
    path=ROOT/"inputs/manifest.json"
    if path.exists():assert read(path)==out
    else:dump(path,out)

if __name__=="__main__":main()
