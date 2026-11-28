"""Recheck the branch from the per-metric UCM criteria separately from the execution switch."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=="__main__":
    first=root.parent/"first_round/evaluation/results/单次检索.json";scene=root.parent/"scene_evidence/结果.json"
    a=read(first);b=read(scene);gate=read(root/"inputs/gate.json")
    steps=[*a["formal_criteria"].values(),*a["extra164_preserved"].values()]
    assert len(steps)==6 and all(isinstance(v,bool) for v in steps)
    firstpass=all(steps)
    de=b["differences"]
    flags=[de["strict_real-"+c][panel]["mR"][field][0]>0 if field=="interval" else de["strict_real-"+c][panel]["mR"][field]>0
           for c in ("strict_none","strict_shuffled") for panel,field in (("official","interval"),("extra164","mean"))]
    evidence=all(flags)
    branch="first_round" if firstpass else "scene_evidence" if evidence else "skip"
    assert firstpass==a["joint_pass"]==gate["r3_joint_pass"] and evidence==b["evidence_gain"]==gate["evidence_pass"]
    assert gate["branch"]==branch and gate["first_result"]==sha(first) and gate["scene_result"]==sha(scene)
    (root/"audit/按指标独立核对分支.json").write_text(json.dumps(dict(branch=branch,
          first_formal=a["formal_criteria"],first_extra=a["extra164_preserved"],
          evidence_control_criteria=flags,gate=sha(root/"inputs/gate.json")),indent=2)+"\n")
