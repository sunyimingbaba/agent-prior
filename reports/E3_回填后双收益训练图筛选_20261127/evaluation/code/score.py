"""Score actual train arms and matched reference outputs on identical frozen UCM data."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys
import tarfile
import numpy as np
import torch

EVAL = Path(__file__).resolve().parents[1]
STAGE = EVAL.parent
HIST = STAGE.parent / "E3_R3改选收益校准与双批首句验证_20261126/targeted"
sys.path.insert(0, str(HIST/"code"))
from score_final import RFPO, FORM, read, dump, sha, calculate, details, counts, alt_summary, summary_by_seed, paired

ARMS = ("r2","r3","retained","R0")


def pack(tag, folder, hashes):
    archive = EVAL/"archives"/(tag+".tar.gz")
    with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
        for name,h in sorted(hashes.items()):
            assert sha(folder/name)==h
            tar.add(folder/name,arcname=name)
    with tarfile.open(archive,"r|gz") as tar:
        seen=set()
        for member in tar:
            assert member.name not in seen
            seen.add(member.name)
            assert hashlib.file_digest(tar.extractfile(member),"sha256").hexdigest()==hashes[member.name]
        assert seen==set(hashes)
    return dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(hashes))


def inspect_records(folder, rows, expected, inputs):
    proof = read(folder/"complete.json")
    assert sha(folder/"features.pt")==proof["features"]
    bundle = torch.load(folder/"features.pt",map_location="cpu",weights_only=True)
    assert bundle["adapter"]==proof["adapter"]==expected
    names = [r["image"] for r in rows]
    assert bundle["images"]==names and set(proof["images"])==set(names)
    hashes={"complete.json":sha(folder/"complete.json"),"features.pt":proof["features"]}
    meta={}
    for i,r in enumerate(rows):
        n=r["image"]; f=folder/(n+".pt")
        z=torch.load(f,map_location="cpu",weights_only=True)
        hashes[f.name]=proof["images"][n]
        assert sha(f)==hashes[f.name]
        assert z["query"]==r["orig_query"] and z["rgb"]==r["rgb_sha256"] and z["image"]==n
        assert z["protocol"]==proof["protocol"] and z["decoder"]==proof["decoder"]
        assert z["adapter"]==expected and torch.equal(z["v"],bundle["features"][i])
        assert torch.isfinite(z["prior"]).all() and torch.isfinite(z["v"]).all()
        assert inputs.setdefault(n,z["feedback"])==z["feedback"]
        meta[n]=dict(actions=z["actions"],calls=z["calls"],prior=hashlib.sha256(z["prior"].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    return bundle["features"],proof,hashes,meta


def get(seed, arm, gallery, inputs):
    name=seed+"_"+arm
    folder=EVAL/"records"/name
    if arm in ("retained","R0"):
        source=HIST if arm=="retained" else HIST/"start_diagnostic"
        archive_index=read(source/"archives"/("evaluate.json" if arm=="retained" else "index.json"))
        entry=archive_index[name if arm=="retained" else seed]
        assert sha(source/"archives"/(name+".tar.gz"))==entry["sha256"]
        expected=entry["adapter"]
        anchor=[gallery[i] for i in (0,105,209)]
        z,proof,hashes,_=inspect_records(folder,anchor,expected,inputs)
        visual,old,_,meta=inspect_records(source/"evaluate"/name,gallery,expected,inputs)
        assert proof["reference"]==sha(source/"evaluate"/name/"complete.json")
        assert torch.equal(z,visual[[0,105,209]]) and all(proof[k]==old[k] for k in ("torch","device"))
        anchors=pack(name+"_anchors",folder,hashes)
        assets=dict(anchor_archive=anchors,reference_archive=entry)
    else:
        train=read(STAGE/"models"/name/"complete.json")
        roster=STAGE/"scoring/locks/training.json"
        rows=read(roster)["rows"]
        assert 2<=len(rows)<=24 and train["train"]==sha(roster) and train["steps"]==len(rows)
        assert sha(STAGE/"models"/name/"steps.json")==train["scores"]
        assert train["lock"] == sha(STAGE/"training/train_lock.json")
        assert train["initial"] == read(HIST/"start_diagnostic/archives/index.json")[seed]["adapter"]
        steps = read(STAGE/"models"/name/"steps.json")
        train_names = {r["image"] for r in steps}
        assert [r["image"] for r in steps] == [r["image"] for r in rows]
        assert len(train_names) == len(rows) and train_names.isdisjoint({r["image"] for r in gallery})
        assert {r["rgb_sha256"] for r in rows}.isdisjoint({r["rgb_sha256"] for r in gallery})
        expected=train["final"]
        visual,proof,hashes,meta=inspect_records(folder,gallery,expected,inputs)
        assets=pack(name,folder,hashes)
    assert proof["decoder"]==sha(EVAL/"code/predict.py") and proof["protocol"]==sha(EVAL/"验收口径.md")
    return visual, meta, dict(**assets,env={k:proof[k] for k in ("torch","device","cuda","source_hashes")},adapter=expected)


def run():
    gallery=read(FORM/"inputs/ucm.json")
    ids={r["rgb_sha256"]:i for i,r in enumerate(gallery)}
    overlap=[r for r in read(RFPO/"inputs/panel.json") if r["rgb_sha256"] in ids]
    assert len(overlap)==106 and len(ids)==210
    hard=torch.load(RFPO/"records/text.pt",weights_only=True)
    native=torch.load(FORM/"results/ucm_text.pt",weights_only=True)
    cols=native["caption_order"].numpy().reshape(210,5)
    values={key:defaultdict(dict) for key in ("official","first106","alternatives")}
    raw=[];assets={};inputs={};same={}
    for seed in "ab":
        meta={}
        for arm in ARMS:
            visual,meta[arm],assets[seed+"_"+arm]=get(seed,arm,gallery,inputs)
            full=values["official"][arm][seed]=calculate("official",visual,gallery)
            fine=values["first106"][arm][seed]=details(visual,overlap,ids,cols,native,hard)
            owners,hits,denom=counts(visual,native,"alternatives")
            assert owners.tolist()==list(range(210)) and denom.sum()==833
            if not raw:
                capcounts=denom
            else:
                assert np.array_equal(denom,capcounts)
            values["alternatives"][arm][seed]=hits
            per={r["image"]:dict(image=r["image"],seed=seed,arm=arm,official={k:float(v[i]) for k,v in full.items()},
                              alternatives=dict(hits=hits[i].tolist(),captions=int(denom[i])))
                 for i,r in enumerate(gallery)}
            for i,r in enumerate(overlap):
                per[r["image"]]["first106"]={k:float(v[i]) for k,v in fine.items()}
            raw.extend(per.values())
        same[seed]={field:sum(meta["r3"][n][field]==meta["r2"][n][field] for n in meta["r2"])
                    for field in ("actions","calls","prior")}
    assert len({json.dumps(r["env"],sort_keys=True) for r in assets.values()})==1
    contrasts=(("r3","r2"),("r3","retained"),("r2","retained"),("r3","R0"),("r2","R0"))
    result={}
    for key in ("official","first106"):
        unit=[r["image"] for r in gallery] if key=="official" else [r["pair"] for r in overlap]
        val=values[key]
        result[key]=dict(images=len(unit),units=len(set(unit)),levels={a:summary_by_seed(val[a],unit) for a in ARMS},
                         contrasts={a+"-"+b:paired(val[a],val[b],unit) for a,b in contrasts})
        hist=read(HIST/"start_diagnostic/results/冷启动前后对照.json")[key]["levels"]
        assert all(result[key]["levels"][a]==hist[a] for a in ("retained","R0"))
    alt=values["alternatives"]
    result["alternatives"]=dict(images=len(denom),captions=int(denom.sum()),levels={a:alt_summary(alt[a],denom) for a in ARMS},
        contrasts={a+"-"+b:alt_summary({s:alt[a][s]-alt[b][s] for s in "ab"},denom) for a,b in contrasts})
    delta=result["first106"]["contrasts"]["r3-r2"]
    mr=result["official"]["contrasts"]["r3-r2"]["mR"]
    result["criteria"]=dict(detail_lower_positive=delta["DE"]["interval"][0]>0,
                            mR_preserved=mr["mean"]>=0,other_caption_MRR_preserved=delta["mrr"]["mean"]>=0)
    result["passes"]=all(result["criteria"].values())
    result["matched_actions_among_210"]=same
    prev_eval=STAGE.parent/"E3_R2逐候选分值与困难细节加分_20261127/evaluation/results"
    prev_path=prev_eval/"单次检索对照.json"
    assert sha(prev_path)==read(prev_eval/"complete.json")["files"]["results/单次检索对照.json"]
    previous=read(prev_path)
    result["previous_24_image_R3_reference"]={
        "source_sha256":sha(prev_path),
        "official_mR":previous["official"]["levels"]["r3"]["mR"],
        "DE":previous["first106"]["levels"]["r3"]["DE"],
        "other_caption_MRR":previous["first106"]["levels"]["r3"]["mrr"],
    }
    dump(EVAL/"archives/index.json",assets)
    dump(EVAL/"results/逐图指标.json",raw)
    dump(EVAL/"results/单次检索对照.json",result)
    files=("archives/index.json","results/逐图指标.json","results/单次检索对照.json","code/score.py","code/predict.py","验收口径.md")
    dump(EVAL/"results/complete.json",dict(files={f:sha(EVAL/f) for f in files},new_outputs=840,anchor_checks=12,external_reference_outputs=840))


if __name__=="__main__":
    torch.set_num_threads(1)
    run()
