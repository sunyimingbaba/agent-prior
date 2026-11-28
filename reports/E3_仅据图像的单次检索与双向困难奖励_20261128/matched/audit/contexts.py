"""Recheck per-photo raw prefix logs and produce reproducible compressed evidence copies."""
from pathlib import Path
import gzip,hashlib,io,json,tarfile
import torch

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sig(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def pack(tag,paths):
    members={}
    output=ROOT/"archive"/(tag+".tar.gz")
    output.parent.mkdir(exist_ok=True)
    with output.open("wb") as stream,gzip.GzipFile(fileobj=stream,filename="",mode="wb",mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode="w|",format=tarfile.USTAR_FORMAT) as tar:
            for f in sorted(paths):
                name=f.relative_to(ROOT).as_posix()
                data=f.read_bytes();mem=tarfile.TarInfo(name);mem.size=len(data)
                tar.addfile(mem,io.BytesIO(data));members[name]=hashlib.sha256(data).hexdigest()
    with tarfile.open(output) as tar:
        assert {i.name:hashlib.sha256(tar.extractfile(i).read()).hexdigest() for i in tar}==members
    return dict(file=output.name,sha256=sha(output),members=members)


def main():
    lock=read(ROOT/"inputs/train_lock.json"); rows=read(ROOT/"inputs/roster.json")["rows"]
    tool=read(ROOT/"inputs/feedback_verified.json")
    assert tool["lock"]==sha(ROOT/"inputs/train_lock.json")
    templates={}; comparison={}; output={}
    for seed in "ab":
        c=ROOT/"contexts"/seed;proof=read(c/"complete.json")
        assert proof["lock"]==tool["lock"] and proof["feedback"]==sha(ROOT/"inputs/feedback_verified.json")
        assert set(proof["files"])=={r["image"] for r in rows} and proof["adapter"]==lock["starts"][seed]
        reviewed={};eligible=[]
        for row in rows:
            n=row["image"];f=c/(n+".pt");assert sha(f)==proof["files"][n]
            data=torch.load(f,map_location="cpu",weights_only=True)
            assert data["rgb"]==row["rgb_sha256"]==tool["images"][n]["rgb"] and data["image"]==n
            assert data["row"]==row["row"] and data["adapter"]==lock["starts"][seed]
            assert data["query"]==lock["query"] and data["template"].count(lock["query"])==1
            assert hashlib.sha256(data["template"].encode()).hexdigest()==proof["template"]
            assert data["token_hash"]==sig(data["first_input_ids"]) and len(data["first"])>0
            assert data["eligible"]==bool(data["calls"])==proof["eligible"][n]
            first=(data["token_hash"],data["pixel_values_sha"],tuple(data["grid"].flatten().tolist()))
            assert comparison.setdefault(n,first)==first
            if data["eligible"]:
                assert data["calls"]==[tool["images"][n]["feedback"]] and data["answer"]==data["calls"][0]
                q=data["first_input_ids"].shape[-1];assert torch.equal(data["first_input_ids"],data["prefix"][:,:q])
                assert data["prefix"].shape[-1]>q
                eligible.append(n)
            else:assert data["prefix"] is None and data["answer"] is None
            reviewed[n]=dict(first_tokens=len(data["first"]),tool_called=data["eligible"],
                             input_ids=data["token_hash"],pixel_values=data["pixel_values_sha"])
        assert len(eligible)==proof["available"]
        templates[seed]=proof["template"];output[seed]=dict(reached=len(eligible),observed=reviewed)
    assert len(set(templates.values()))==1
    save(ROOT/"audit/固定问题真实输入核查.json",dict(seeds=output,canonical_template=templates["a"],query=lock["query"]))
    states=list((ROOT/"models").glob("*/trajectories/*.pt"))
    expected=2*sum(s["reached"] for s in output.values())
    assert len(states)==expected
    first=pack("first_rounds",list((ROOT/"contexts").glob("*/*.pt"))+list((ROOT/"contexts").glob("*/complete.json")))
    generated=pack("training_trajectories",states)
    save(ROOT/"archive/SHA256.json",dict(first_rounds=first,training=generated,groups=expected))
if __name__=="__main__":main()
