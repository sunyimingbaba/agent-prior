"""Compute the preregistered visual and scene baselines with a second tie-aware implementation."""
from pathlib import Path
import gzip,importlib.util,json,hashlib,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];SEM=ROOT.parent;STAGE=SEM.parent
CROSS=STAGE/"cross_replication";FORM=STAGE.parent/"E3-2_UCM与RSITMD独立复验_20261124"
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def save(p,z):p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n")
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module(CROSS/"code/score_cross.py","previous_cross")
previous=base.previous

def capture(tag,source,rows,lock,gate):
    folder=source/"records"/tag;proof=read(folder/"complete.json");weights=gate["policies"][tag]
    assert proof["gate"]==sha(source/"inputs/gate.json") and proof["verified"]==sha(source/"inputs/verified.json")
    assert proof["adapter"]==weights["adapter"] and proof["code"]==lock["cross_generator"]
    tf=folder/"features.pt";assert sha(tf)==proof["features"]
    z=torch.load(tf,map_location="cpu",weights_only=True)
    assert z["images"]==[r["image"] for r in rows] and z["features"].shape==(448,768)
    files={"features.pt":proof["features"],"complete.json":sha(folder/"complete.json")}
    for i,r in enumerate(rows):
        n=r["image"];f=folder/(n+".pt"); assert sha(f)==proof["images"][n]
        rec=torch.load(f,weights_only=True,map_location="cpu")
        assert (rec["image"],rec["rgb"],rec["query"],rec["feedback"],rec["evidence_rgb"],rec["adapter"])==(n,r["rgb"],lock["query"],r["feedback"],None,weights["adapter"])
        assert rec["actions"] and all(t==r["feedback"] for t in rec["calls"])
        assert torch.equal(rec["v"],z["features"][i]) and torch.isfinite(rec["prior"]).all()
        files[f.name]=proof["images"][n]
    if source==ROOT:
        base.ROOT=ROOT
        arch=base.archive(tag,folder,files)
    else:
        old=read(source/"archive/index.json")[tag]
        assert sha(source/"archive"/old["file"])==old["sha256"]
        arch=dict(reused_directory=str(source.relative_to(STAGE)),**old)
    return z["features"],arch


def main():
    gate=read(ROOT/"inputs/gate.json");lock=read(ROOT/"inputs/manifest.json")
    assert gate["lock"]==sha(ROOT/"inputs/manifest.json") and gate["ucm_result"]==sha(SEM/"结果.json")
    if not gate["system_gain"]:
        save(ROOT/"results/跨库融合.json",dict(status="skipped",weight_image_generations=0,system_replication=None,
             gate=sha(ROOT/"inputs/gate.json")))
        return
    assert gate["fusion_method"]==sha(SEM/"fusion.py") and gate["source_bank"]==sha(SEM/"场景类别向量.pt")
    assert read(SEM/"原生文本编码校验.json")["class_features"]==gate["source_bank"]
    assert all(sha(ROOT/f)==h for f,h in lock["files"].items())
    gallery=FORM/"results/rsitmd_text.pt";assert sha(gallery)==lock["external"]["gallery_text"]
    bank=torch.load(gallery,weights_only=True,map_location="cpu")
    class_bank=torch.load(SEM/"场景类别向量.pt",weights_only=True,map_location="cpu")
    assert class_bank["features"].shape==(21,768) and len(set(class_bank["classes"]))==21
    fit=module(SEM/"fusion.py","provided_fusion")
    rows=read(ROOT/"inputs/roster.json");donors=read(ROOT/"inputs/donors.json")
    src=CROSS if gate["first_priors_reused"] else ROOT
    verified=read(src/"inputs/verified.json")
    own=[];shuffled=[]
    for r in rows:
        assert r["feedback"]==verified["items"][r["rgb"]]["feedback"]
        own.append(fit.scene_vector(r["feedback"],class_bank))
        shuffled.append(fit.scene_vector(verified["items"][donors[r["rgb"]]]["feedback"],class_bank))
    own=torch.stack(own);shuffled=torch.stack(shuffled)
    assert own.shape==shuffled.shape==(448,768)
    results={};captured={};raw=[]
    def measure(key,v):
        ids,h,ds=previous.counts(v,bank,"native")
        ids2,h2,ds2=previous.independent(v,bank,"native")
        np.testing.assert_array_equal(ids,ids2);np.testing.assert_array_equal(ds,ds2)
        np.testing.assert_allclose(h,h2,rtol=0,atol=1e-12)
        assert len(ds)==448 and (ds==5).all()
        raw.extend(dict(tag=key,image=r["image"],hits=x.tolist()) for r,x in zip(rows,h))
        return h
    scene_hits=measure("only_scene",own)
    denom=np.full(448,5)
    for tag in gate["tags"]:
        v,captured[tag]=capture(tag,src,rows,lock,gate)
        for condition,arr in (("none",v),("real",fit.fuse(v,own)),("shuffled",fit.fuse(v,shuffled))):
            results[tag.removesuffix("none")+condition]=measure(tag.removesuffix("none")+condition,arr)
    def report(x):
        z=previous.summary(x.mean(0),denom)
        for i,k in enumerate(previous.KEYS):
            z[k]["seeds"]={s:float(previous.scalar_values(q.sum(0),np.asarray(448),np.asarray(2240))[i]) for s,q in zip("ab",x)}
        return z
    arms={};contrasts={}
    for arm in ("strict","r2"):
        for condition in ("none","real","shuffled"):
            name=arm+"_"+condition
            arms[name]=report(np.stack([results[s+"_"+name] for s in "ab"]))
    criteria=[]
    for x,y in [(r+"_real",r+"_"+c) for r in ("strict","r2") for c in ("none","shuffled")]+[
          ("strict_"+c,"r2_"+c) for c in ("none","real","shuffled")]:
        arr=np.stack([results[s+"_"+x]-results[s+"_"+y] for s in "ab"])
        z=report(arr);previous.independent_change(arr.mean(0),denom,z)
        contrasts[x+"-"+y]=z
        if x=="strict_real" and y in ("strict_none","strict_shuffled"):criteria.append(z["mR"]["interval"][0]>0)
    assert len(criteria)==2
    out=dict(status="completed",images=448,captions=2240,underlying_weight_image_outputs=1792,
             new_weight_image_generations=0 if gate["first_priors_reused"] else 1792,reused=gate["first_priors_reused"],
             category_text_bank=gate["source_bank"],levels=arms,only_scene=previous.summary(scene_hits,denom),
             contrasts=contrasts,system_replication=all(criteria),ucm_gate=sha(ROOT/"inputs/gate.json"))
    save(ROOT/"results/跨库融合.json",out);save(ROOT/"archive/index.json",captured)
    with gzip.open(ROOT/"results/逐图召回.jsonl.gz","wt") as f:
        for row in raw:f.write(json.dumps(row)+"\n")
    files=["inputs/gate.json","inputs/manifest.json","archive/index.json","results/跨库融合.json","results/逐图召回.jsonl.gz"]
    save(ROOT/"results/complete.json",dict(files={f:sha(ROOT/f) for f in files},verified_gallery_per_condition=len(rows)))
if __name__=="__main__":
    torch.set_num_threads(1);main()
