"""Use the UCM fusion gate and resolve any previously verified identical raw image priors."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEM=ROOT.parent;STAGE=SEM.parent;CROSS=STAGE/"cross_replication"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):return json.loads(p.read_bytes())
def main():
    spec=read(ROOT/"inputs/manifest.json")
    assert all(sha(ROOT/n)==h for n,h in spec["files"].items())
    assert spec["cross_plan"]==sha(CROSS/"inputs/manifest.json") and spec["previous_protocol"]==sha(SEM/"事前口径.md")
    res=SEM/"结果.json";u=read(res);approved=u["system_gain"];assert isinstance(approved,bool)
    previous=read(CROSS/"inputs/gate.json"); assert previous["lock"]==spec["cross_plan"]
    policies={}
    for s in "ab":
        for a in ("r2","strict"):
            orig=STAGE/"first_round/models"/(s+"_"+a);f=orig/"complete.json";q=read(f)
            assert q["steps"]==24 and q["lock"]==spec["first_train"] and q["scores"]==sha(orig/"steps.json")
            policies[s+"_"+a+"_none"]=dict(adapter=q["final"],initial=q["initial"],proof=sha(f))
    reuse=approved and previous["branch"]=="first_round"
    if reuse:
        assert set(policies)==set(previous["policies"])
        assert all(p["adapter"]==previous["policies"][n]["adapter"] and p["proof"]==previous["policies"][n]["proof"] for n,p in policies.items())
    bank=SEM/"场景类别向量.pt";fn=SEM/"fusion.py"
    assert bank.is_file() and fn.is_file()
    g=dict(branch="first_round" if approved else "skip",system_gain=approved,previous_cross_gate=sha(CROSS/"inputs/gate.json"),
           first_priors_reused=reuse,ucm_result=sha(res),source_bank=sha(bank),fusion_method=sha(fn),
           tags=list(policies) if approved else [],policies=policies if approved else {},lock=sha(ROOT/"inputs/manifest.json"))
    target=ROOT/"inputs/gate.json"
    if target.exists():assert read(target)==g
    else:target.write_text(json.dumps(g,ensure_ascii=False,indent=2)+"\n")
    print(g["branch"],"reuse=",reuse)
if __name__=="__main__":main()
