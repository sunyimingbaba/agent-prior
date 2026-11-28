"""Recompute both directions in two ways, then resample paired images on the fixed gallery."""
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT.parent
OLD=STAGE.parent/"E3_视觉细节复核与RFPO_20261127/cross448"
FORM=STAGE.parent/"E3-2_UCM与RSITMD独立复验_20261124"
sys.path.insert(0,str(OLD/"code"))
import score as previous
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def archive(tag,folder,files):
    dst=ROOT/"archive"/(tag+".tar.gz")
    with dst.open("wb") as output,gzip.GzipFile(fileobj=output,filename="",mode="wb",mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode="w|",format=tarfile.USTAR_FORMAT) as tape:
            for n,h in sorted(files.items()):
                b=(folder/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h
                info=tarfile.TarInfo(n);info.size=len(b);tape.addfile(info,io.BytesIO(b))
    with tarfile.open(dst) as tape:
        assert {t.name:hashlib.sha256(tape.extractfile(t).read()).hexdigest() for t in tape}==files
    return dict(file=dst.name,sha256=sha(dst),members=len(files))


def main():
    spec=read(ROOT/"inputs/manifest.json");gate=read(ROOT/"inputs/gate.json");branch=gate["branch"]
    assert gate["lock"]==sha(ROOT/"inputs/manifest.json")
    if branch=="skip":
        save(ROOT/"results/条件复核.json",dict(branch=branch,inferences=0,ucm_gate=sha(ROOT/"inputs/gate.json"),cross_domain_replication=None))
        return
    assert sha(OLD/"code/score.py")==spec["external"]["old448_score"]
    f=FORM/"results/rsitmd_text.pt";assert sha(f)==spec["external"]["gallery_text"]
    bank=torch.load(f,weights_only=True,map_location="cpu")
    roster=read(ROOT/"inputs/roster.json");donors=read(ROOT/"inputs/donors.json")
    fb=read(ROOT/"inputs/verified.json");assert fb["gate"]==sha(ROOT/"inputs/gate.json")
    records={};archive_index={};features={};raw=[]
    for tag in gate["tags"]:
        folder=ROOT/"records"/tag;proof=read(folder/"complete.json")
        assert proof["gate"]==sha(ROOT/"inputs/gate.json") and proof["verified"]==sha(ROOT/"inputs/verified.json")
        assert proof["adapter"]==gate["policies"][tag]["adapter"] and proof["code"]==sha(ROOT/"code/generate.py")
        z=torch.load(folder/"features.pt",weights_only=True,map_location="cpu")
        assert sha(folder/"features.pt")==proof["features"] and z["images"]==[r["image"] for r in roster]
        assert len(proof["images"])==448; files={"features.pt":proof["features"],"complete.json":sha(folder/"complete.json")}
        for i,r in enumerate(roster):
            n=r["image"];f=folder/(n+".pt");assert sha(f)==proof["images"][n]
            x=torch.load(f,map_location="cpu",weights_only=True);cond=tag.split("_")[2]
            donor=None if cond=="none" else donors[r["rgb"]] if cond=="shuffled" else r["rgb"]
            question=spec["query"] if donor is None else spec["query"]+"\n"+spec["label"]+fb["items"][donor]["feedback"]
            assert (x["image"],x["rgb"],x["adapter"],x["evidence_rgb"],x["query"],x["feedback"])==(n,r["rgb"],proof["adapter"],donor,question,r["feedback"])
            assert x["actions"] and all(v==x["feedback"] for v in x["calls"])
            assert torch.equal(x["v"],z["features"][i]) and torch.isfinite(x["prior"]).all() and len(x["tokens"])==64
            files[f.name]=proof["images"][n]
        ids,h,ds=previous.counts(z["features"],bank,"native")
        ids2,h2,ds2=previous.independent(z["features"],bank,"native")
        np.testing.assert_array_equal(ids,ids2);np.testing.assert_array_equal(ds,ds2)
        np.testing.assert_allclose(h,h2,rtol=0,atol=1e-12)
        assert len(ds)==448 and (ds==5).all()
        records[tag]=h;archive_index[tag]=archive(tag,folder,files)
        raw.extend(dict(tag=tag,image=r["image"],hits=v.tolist()) for r,v in zip(roster,h))
    def summary(values):
        ci=previous.summary(values.mean(0),ds)
        seeds=[previous.scalar_values(x.sum(0),np.asarray(len(ds)),np.asarray(ds.sum())) for x in values]
        for i,key in enumerate(previous.KEYS):ci[key]["seeds"]={s:float(v[i]) for s,v in zip("ab",seeds)}
        return ci
    arms=sorted({tag.split("_",1)[1] for tag in gate["tags"]}); stats={};comparisons={}
    for arm in arms:stats[arm]=summary(np.stack([records[s+"_"+arm] for s in "ab"]))
    pairs=[("strict_none","r2_none")] if branch=="first_round" else [("strict_real","strict_shuffled"),("strict_real","strict_none")]
    for a,b in pairs:
        x=np.stack([records[s+"_"+a]-records[s+"_"+b] for s in "ab"])
        entry=summary(x);previous.independent_change(x.mean(0),ds,entry);comparisons[a+"-"+b]=entry
    result=dict(branch=branch,images=448,captions=2240,inferences=448*len(arms)*2,
         ucmbasis=sha(ROOT/"inputs/gate.json"),arms=stats,contrasts=comparisons,
         cross_domain_replication=all(d["mR"]["interval"][0]>0 for d in comparisons.values()))
    save(ROOT/"results/条件复核.json",result)
    save(ROOT/"archive/index.json",archive_index)
    with gzip.open(ROOT/"results/逐图召回.jsonl.gz","wt") as f:
        for row in raw:f.write(json.dumps(row)+"\n")
    paths=["inputs/gate.json","inputs/manifest.json","inputs/verified.json","archive/index.json","results/条件复核.json","results/逐图召回.jsonl.gz"]
    save(ROOT/"results/complete.json",dict(files={f:sha(ROOT/f) for f in paths},inferences=result["inferences"]))
if __name__=="__main__":
    torch.set_num_threads(1);main()
