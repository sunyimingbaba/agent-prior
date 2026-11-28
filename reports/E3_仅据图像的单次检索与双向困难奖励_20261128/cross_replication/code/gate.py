"""Select one transfer branch solely from prespecified local UCM results."""
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=ROOT/"inputs/manifest.json";spec=read(lock)
    assert all(sha(ROOT/f)==h for f,h in spec["files"].items())
    first=ROOT.parent/"first_round/evaluation";scene=ROOT.parent/"scene_evidence"
    name="results/单次检索.json";f=first/name;e=scene/"结果.json"
    assert sha(f)==read(first/"results/complete.json")["files"][name]
    x=read(f);s=read(e)
    assert isinstance(x["joint_pass"],bool) and isinstance(s["evidence_gain"],bool)
    assert x["joint_pass"]==(x["formal_pass"] and x["extra164_preserved"])
    diffs=s["differences"]
    expect=all(diffs["strict_real-"+c]["official"]["mR"]["interval"][0]>0 and
               diffs["strict_real-"+c]["extra164"]["mR"]["mean"]>0 for c in ("strict_none","strict_shuffled"))
    assert expect==s["evidence_gain"]
    firstpass=x["joint_pass"];scene_pass=s["evidence_gain"]
    tags=[];policies={}
    if firstpass:
        branch="first_round"
        configs=[(seed,arm,"none") for seed in "ab" for arm in ("r2","strict")]
    elif scene_pass:
        branch="scene_evidence"
        configs=[(seed,"strict",c) for seed in "ab" for c in ("real","shuffled","none")]
    else:
        branch="skip";configs=[]
    for seed,arm,c in configs:
        name=seed+"_"+arm;tag=name+"_"+c
        if firstpass:
            folder=first.parent/"models"/name;cp=read(folder/"complete.json")
            assert cp["steps"]==24 and cp["scores"]==sha(folder/"steps.json")
            pol=dict(adapter=cp["final"],proof=sha(folder/"complete.json"),initial=cp["initial"])
        else:pol=spec["old_policies"][name]
        tags.append(tag);policies[tag]=pol
    out=dict(branch=branch,r3_joint_pass=firstpass,evidence_pass=scene_pass,first_result=sha(f),scene_result=sha(e),
             first_complete=sha(first/"results/complete.json"),scene_trajectories=sha(scene/"轨迹清单.json"),
             lock=sha(lock),tags=tags,policies=policies)
    dst=ROOT/"inputs/gate.json"
    if dst.exists():assert out==read(dst)
    else:dst.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(branch)
if __name__=="__main__":main()
