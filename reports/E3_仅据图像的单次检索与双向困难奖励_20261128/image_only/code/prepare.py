from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parents[1]
OLD=REPORTS/"E3_视觉细节复核与RFPO_20261127"
POOL=REPORTS/"E3_回填后双收益训练图筛选_20261127"
FORM=REPORTS/"E3-2_UCM与RSITMD独立复验_20261124"
FINE=REPORTS/"E3_UCM多描述奖励同预算RFPO_20261124"
QUERY="Describe the visible features of this remote sensing image."
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):return json.loads(p.read_bytes())
def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def create(p,x):
    if p.exists():assert read(p)==x
    else:dump(p,x)
def main():
    official=read(FORM/"inputs/ucm.json"); extra=read(OLD/"extra164/inputs/rows.json"); fb=read(FORM/"records/tools.json")
    assert sha(FORM/"records/tools.json")==read(FORM/"records/environment.json")["tools"]
    assert sha(FORM/"inputs/ucm.json")==read(FORM/"inputs/audit.json")["regions"]["ucm"]["manifest"]
    assert sha(OLD/"extra164/inputs/rows.json")==read(OLD/"extra164/inputs/manifest.json")["rows"]
    out=[]
    for panel,rows in (("official",official),("extra",extra)):
        for r in rows:
            assert QUERY not in r["captions"]
            out.append(dict(panel=panel,image=r["image"],split=r.get("split","test"),row=r["row"],rgb=r["rgb_sha256"],
                feedback=fb[r["rgb_sha256"]][0] if panel=="official" else r["feedback"]))
    pixels={r["rgb"] for r in out}; train=read(POOL/"scoring/locks/training.json")["rows"]
    assert len(official)==210 and len(extra)==164 and len(pixels)==374 and len(train)==24 and pixels.isdisjoint(r["rgb_sha256"] for r in train)
    create(ROOT/"inputs/roster.json",out)
    policies={}
    for s in "ab":
        for a,p in (("r2",POOL),("strict",OLD)):
            f=p/"models"/(s+"_"+a)/"complete.json";cp=read(f)
            assert cp["steps"]==24 and cp["scores"]==sha(f.parent/"steps.json") and cp["train"]==sha(POOL/"scoring/locks/training.json")
            policies[s+"_"+a]=dict(adapter=cp["final"],initial=cp["initial"],proof=sha(f),steps=cp["steps"])
        assert policies[s+"_r2"]["initial"]==policies[s+"_strict"]["initial"]
    inputs={str(p.relative_to(REPORTS)):sha(p) for p in (FORM/"results/ucm_text.pt",FINE/"inputs/panel.json",FINE/"records/text.pt",FORM/"inputs/ucm.json",
           OLD/"extra164/inputs/rows.json",OLD/"extra164/inputs/texts.pt",OLD/"evaluation/results/单次检索对照.json",OLD/"extra164/summary.json")}
    files=["口径.md","code/prepare.py","code/run.sh","code/predict.py","inputs/roster.json"]
    create(ROOT/"inputs/manifest.json",dict(query=QUERY,policies=policies,files={f:sha(ROOT/f) for f in files},score_inputs=inputs,
           scene_head=read(FORM/"records/environment.json")["scene_head"],official=210,extra=164,train=24,overlap=0))
if __name__=="__main__":main()
