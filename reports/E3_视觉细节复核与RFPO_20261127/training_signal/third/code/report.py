"""Fixed paired bootstrap and two independent copies of the complete underlying records."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def archive(label, paths):
    name = ROOT/"archive"/(label+".tar.gz")
    name.parent.mkdir(exist_ok=True)
    expected = {}
    with name.open("wb") as remote, gzip.GzipFile(fileobj=remote,filename="",mode="wb",mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode="w|",format=tarfile.USTAR_FORMAT) as tape:
            for path in sorted(paths):
                payload=path.read_bytes()
                relative=path.relative_to(ROOT).as_posix(); expected[relative]=sha(path)
                entry=tarfile.TarInfo(relative);entry.size=len(payload)
                tape.addfile(entry,io.BytesIO(payload))
    with tarfile.open(name) as t:
        assert {x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest() for x in t}==expected
    return dict(file=name.name,sha256=sha(name),members=expected)


def main():
    lock=read(ROOT/"inputs/roster_lock.json")
    assert sha(ROOT/"事前口径.md")==lock["protocol"] and all(sha(ROOT/"code"/n)==v for n,v in lock["scripts"].items())
    assert sha(ROOT/"inputs/roster.json")==lock["roster"] and sha(ROOT/"inputs/groups.json")==lock["groups"]
    raw=read(ROOT/"e_complete.json"); bank=read(ROOT/"scoring_complete.json"); feed=read(ROOT/"feedback_verified.json")
    assert bank["sampling"]==sha(ROOT/"e_complete.json") and bank["inputs"]==sha(ROOT/"inputs/roster_lock.json")
    assert raw["adapter"]==lock["model"] and raw["lock"]["roster"]==feed["roster"]==lock["roster"]
    roster=read(ROOT/"inputs/roster.json");groups={};records=[]
    for row in roster:
        n=row["image"];sp=ROOT/"scores"/(n+".json");rp=ROOT/"raw/e"/(n+".pt")
        assert sha(sp)==bank["files"][n] and sha(rp)==raw["files"][n]
        v=read(sp); tr=torch.load(rp,weights_only=True,map_location="cpu");trace=tr["eligible"]
        assert v["raw_sha256"]==sha(rp) and tr["rgb"]==v["rgb_sha256"]==row["rgb_sha256"]==feed["images"][n]["rgb"]
        assert tr["query"]==row["orig_query"]==row["captions"][0] and tr["feedback"]==row["feedback"]==feed["images"][n]["feedback"]
        assert tr["adapter"]==raw["adapter"] and tr["rng"]==lock["seed_base"]+1000*row["row"] and tr["seed"]=="e"
        assert trace==v["eligible"]==raw["eligibility"][n] and tr["first"]
        if trace:
            assert tr["hidden"].shape==(8,3584) and torch.isfinite(tr["hidden"]).all()
            assert len(tr["paths"])==len(tr["extras"])==len(tr["logps"])==len(tr["texts"])==8
            assert all(0<len(path)==len(lp)<=64 for path,lp in zip(tr["paths"],tr["logps"]))
            assert tr["context"]["first"]==tr["first"] and tr["context"]["answer"]==tr["feedback"] and tr["context"]["prefix"].ndim==2
        compact={k:v[k] for k in ("image","cohort","pair","attribute","side","selected","eligible","replay")}
        records.append(compact)
        groups.setdefault((v["cohort"],v["pair"]),{})[v["selected"]]=v
    ordered=sorted(groups)
    assert len(ordered)==12 and len(records)==24
    x=np.array([[int(groups[k][b]["replay"]["changed"] is True) for b in (True,False)] for k in ordered])
    differences=x[:,0]-x[:,1]
    rng=np.random.default_rng(lock["bootstrap_seed"])
    bs=differences[rng.integers(0,12,(20000,12))].mean(1)
    eligible=np.array([all(g["eligible"] for g in groups[k].values()) for k in ordered])
    qr=np.quantile(bs,[.025,.975])
    counts={str(i)+str(j):int(((x[:,0]==i)&(x[:,1]==j)).sum()) for i in (0,1) for j in (0,1)}
    summary=dict(pairs=12,draws=24,eligible=int(sum(raw["eligibility"].values())),reached_pairs=int(eligible.sum()),
         selected_changed=int(x[:,0].sum()),partner_changed=int(x[:,1].sum()),selected_rate=float(x[:,0].mean()),
         partner_rate=float(x[:,1].mean()),paired_difference=float(differences.mean()),paired_ci95=qr.tolist(),
         pair_counts_selected_partner=counts,eligible_only=dict(pairs=int(eligible.sum()),rate=x[eligible].mean(0).tolist()) if eligible.any() else None,
         inputs=sha(ROOT/"inputs/roster_lock.json"),sampling=sha(ROOT/"e_complete.json"),scoring=sha(ROOT/"scoring_complete.json"))
    save(ROOT/"结果.json",summary);save(ROOT/"逐图首选.json",sorted(records,key=lambda r:(r["cohort"],r["pair"],-r["selected"])))
    raws=archive("trajectories",list((ROOT/"raw/e").glob("*.pt")));scores=archive("scores",list((ROOT/"scores").glob("*.json")))
    assert len(raws["members"])==len(scores["members"])==24
    save(ROOT/"archive/SHA256.json",dict(raw=raws,scores=scores,summary=sha(ROOT/"结果.json"),per_image=sha(ROOT/"逐图首选.json")))
    print(summary)


if __name__=="__main__":main()
