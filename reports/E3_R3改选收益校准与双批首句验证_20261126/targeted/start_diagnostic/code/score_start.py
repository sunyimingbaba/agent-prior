"""Account for the R2-to-R3 gap by evaluating the original cold checkpoint."""
from pathlib import Path
import hashlib
import tarfile
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT.parent
sys.path.insert(0,str(TARGET/"code"))
from score_final import ARMS, read, sha, dump, FORM, RFPO, calculate, details, summary_by_seed, paired


def verify_and_pack(seed, gallery):
    folder = ROOT/"evaluate"/(seed+"_R0")
    p = folder/"features.pt"
    proof = read(folder/"complete.json")
    bundle = torch.load(p,map_location="cpu",weights_only=True)
    saved = read(TARGET/"inputs/pilot/decision.json")["starts"][seed]
    assert sha(p) == proof["features"] and bundle["adapter"] == proof["adapter"] == saved
    assert [r["image"] for r in gallery] == bundle["images"] and len(proof["images"]) == 210
    assert proof["decoder"] == sha(ROOT/"code/greedy_start.py") and proof["protocol"] == sha(ROOT/"诊断说明.md")
    ref = TARGET/"evaluate"/(seed+"_retained")
    formal = read(ref/"complete.json")
    assert proof["formal_eval_code"] == formal["decoder"]
    assert (proof["torch"],proof["device"]) == (formal["torch"],formal["device"])
    members = {"features.pt":sha(p),"complete.json":sha(folder/"complete.json")}
    for i,row in enumerate(gallery):
        n=row["image"]; path=folder/(n+".pt")
        assert sha(path) == proof["images"][n]
        a=torch.load(path,map_location="cpu",weights_only=True)
        rp=ref/path.name
        assert sha(rp) == formal["images"][n]
        b=torch.load(rp,map_location="cpu",weights_only=True)
        assert all(a[k]==b[k] for k in ("image","rgb","query","feedback"))
        assert a["adapter"]==saved and a["decoder"]==proof["decoder"] and a["protocol"]==proof["protocol"]
        assert torch.equal(a["v"],bundle["features"][i]) and torch.isfinite(a["prior"]).all()
        members[path.name]=proof["images"][n]
    archive = ROOT/"archives"/(seed+"_R0.tar.gz")
    with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
        for name in sorted(members):
            tar.add(folder/name,arcname=name)
    with tarfile.open(archive,"r|gz") as tar:
        seen=set()
        for item in tar:
            assert item.name not in seen
            seen.add(item.name)
            assert hashlib.file_digest(tar.extractfile(item),"sha256").hexdigest()==members[item.name]
        assert seen==set(members)
    return bundle["features"],dict(sha256=sha(archive),adapter=saved,files=len(members),bytes=archive.stat().st_size)


def main():
    rawfile=TARGET/"results/逐图检索与细节.json"
    assert sha(rawfile) == read(TARGET/"results/eval.complete.json")["files"]["results/逐图检索与细节.json"]
    raw=read(rawfile)
    existing={(r["seed"],r["arm"],r["image"]):r for r in raw}
    assert len(existing)==len(raw)==1680
    gallery=read(FORM/"inputs/ucm.json")
    ids={r["rgb_sha256"]:i for i,r in enumerate(gallery)}
    overlap=[r for r in read(RFPO/"inputs/panel.json") if r["rgb_sha256"] in ids]
    assert len(overlap)==106
    native=torch.load(FORM/"results/ucm_text.pt",weights_only=True)
    hard=torch.load(RFPO/"records/text.pt",weights_only=True)
    data={tag:{a:{} for a in ("R0",*ARMS)} for tag in ("official","first106")}
    by_image=[]
    archives={}
    for seed in "ab":
        visual,archives[seed]=verify_and_pack(seed,gallery)
        data["official"]["R0"][seed]=calculate("official",visual,gallery)
        data["first106"]["R0"][seed]=details(visual,overlap,ids,native["caption_order"].numpy().reshape(210,5),native,hard)
        for tag,rows in (("official",gallery),("first106",overlap)):
            measured=data[tag]["R0"][seed]
            for j,row in enumerate(rows):
                by_image.append(dict(seed=seed,tag=tag,image=row["image"],values={k:float(v[j]) for k,v in measured.items()}))
            for arm in ARMS:
                data[tag][arm][seed]={k:np.array([existing[seed,arm,r["image"]][tag][k] for r in rows]) for k in measured}
    out={}
    standard=read(TARGET/"results/单次检索对照.json")
    for tag,labels in (("official",[r["image"] for r in gallery]),("first106",[r["pair"] for r in overlap])):
        values=data[tag]
        levels={a:summary_by_seed(values[a],labels) for a in values}
        for a in ARMS:
            assert levels[a]==standard[tag]["levels"][a]
        out[tag]={"images":len(labels), "independent_units":len(set(labels)), "levels":levels,
                  "vs_R0":{a:paired(values[a],values["R0"],labels) for a in ARMS}}
    dump(ROOT/"archives/index.json",archives)
    dump(ROOT/"results/冷启动逐图.json",by_image)
    dump(ROOT/"results/冷启动前后对照.json",out)
    files=("archives/index.json","results/冷启动逐图.json","results/冷启动前后对照.json","code/score_start.py","诊断说明.md")
    dump(ROOT/"results/complete.json",{"observations":420,"files":{f:sha(ROOT/f) for f in files},"formal_inputs":sha(rawfile)})


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
