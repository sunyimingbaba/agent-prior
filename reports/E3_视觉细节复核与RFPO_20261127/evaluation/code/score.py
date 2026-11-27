"""Compare two trained policies with the two matched historical controls."""
from collections import defaultdict
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json
import sys

import numpy as np
import torch

EVAL = Path(__file__).resolve().parents[1]
STAGE = EVAL.parent
POOL = STAGE.parent/"E3_回填后双收益训练图筛选_20261127"
spec = spec_from_file_location("pool_score", POOL/"evaluation/code/score.py")
pool = module_from_spec(spec)
spec.loader.exec_module(pool)
pool.EVAL = EVAL
from score_final import RFPO, FORM, read, dump, sha, calculate, details, counts, alt_summary, summary_by_seed, paired

ANCHOR_IDS = [0, 105, 209]
CONTROLS = ("r2","r3")

def get(seed, arm, gallery, inputs):
    name = f"{seed}_{arm}"
    reference = arm in CONTROLS
    trained = POOL if reference else STAGE
    source = trained/"models"/name
    folder = EVAL/"records"/name
    cp = read(source/"complete.json")
    roster_path = POOL/"scoring/locks/training.json"
    rows = read(roster_path)["rows"]
    lock_path = POOL/"training/train_lock.json" if reference else STAGE/"training"/(arm+"_lock.json")
    initial = read(POOL/"training/train_lock.json")["historical"][seed]["R0/adapter/adapter_model.safetensors"]
    assert cp["steps"]==len(rows)==24 and cp["train"]==sha(roster_path)
    assert cp["initial"]==initial and cp["lock"]==sha(lock_path) and cp["scores"]==sha(source/"steps.json")
    assert [r["image"] for r in read(source/"steps.json")]==[r["image"] for r in rows]
    assert {r["rgb_sha256"] for r in rows}.isdisjoint(r["rgb_sha256"] for r in gallery)
    subset = [gallery[i] for i in ANCHOR_IDS] if reference else gallery
    v,proof,hashes,meta = pool.inspect_records(folder,subset,cp["final"],inputs)
    assert proof["decoder"]==sha(EVAL/"code/predict.py") and proof["protocol"]==sha(EVAL/"验收口径.md")
    filename = name+"_anchors" if reference else name
    assets = dict(current_archive=pool.pack(filename,folder,hashes), adapter=cp["final"],
                  env={k:proof[k] for k in ("torch","device","cuda","source_hashes")})
    if reference:
        prev = POOL/"evaluation"
        index = read(prev/"archives/index.json")
        assert sha(prev/"archives"/(name+".tar.gz"))==index[name]["archive_sha256"]
        full = prev/"records"/name
        assert proof["reference"]==sha(full/"complete.json")
        fullv, old, _, oldmeta = pool.inspect_records(full,gallery,cp["final"],inputs)
        assert torch.equal(v,fullv[ANCHOR_IDS])
        assert all(old[k]==v for k,v in assets["env"].items())
        for r in subset:
            n=r["image"]
            assert meta[n]==oldmeta[n]
        assets["reference_archive"] = index[name]["archive_sha256"]
        v,meta = fullv,oldmeta
    return v,meta,assets

def run(challenger):
    arms=(*CONTROLS,challenger)
    contrasts=[(challenger,control) for control in CONTROLS]
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
        for arm in arms:
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
        same[seed]={other:{field:sum(meta[challenger][n][field]==meta[other][n][field] for n in meta[other])
                          for field in ("actions","calls","prior")} for other in CONTROLS}
    assert len({json.dumps(r["env"],sort_keys=True) for r in assets.values()})==1
    result={}
    for key in ("official","first106"):
        unit=[r["image"] for r in gallery] if key=="official" else [r["pair"] for r in overlap]
        val=values[key]
        result[key]=dict(images=len(unit),units=len(set(unit)),levels={a:summary_by_seed(val[a],unit) for a in arms},
                         contrasts={a+"-"+b:paired(val[a],val[b],unit) for a,b in contrasts})
    alt=values["alternatives"]
    result["alternatives"]=dict(images=len(denom),captions=int(denom.sum()),levels={a:alt_summary(alt[a],denom) for a in arms},
        contrasts={a+"-"+b:alt_summary({s:alt[a][s]-alt[b][s] for s in "ab"},denom) for a,b in contrasts})
    baseline_file=POOL/"evaluation/results/单次检索对照.json"
    assert sha(baseline_file)==read(POOL/"evaluation/results/complete.json")["files"]["results/单次检索对照.json"]
    previous=read(baseline_file)
    for key in ("official","first106","alternatives"):
        assert all(result[key]["levels"][a]==previous[key]["levels"][a] for a in CONTROLS)
    delta=result["first106"]["contrasts"][challenger+"-r2"]
    mr=result["official"]["contrasts"][challenger+"-r2"]["mR"]
    result["criteria"]=dict(detail_lower_positive=delta["DE"]["interval"][0]>0,
                            mR_preserved=mr["mean"]>=0,other_caption_MRR_preserved=delta["mrr"]["mean"]>=0)
    result["passes"]=all(result["criteria"].values())
    result["matched_actions_among_210"]=same
    dump(EVAL/"archives/index.json",assets)
    dump(EVAL/"results/逐图指标.json",raw)
    dump(EVAL/"results/单次检索对照.json",result)
    files=("archives/index.json","results/逐图指标.json","results/单次检索对照.json","code/score.py","code/predict.py","验收口径.md")
    dump(EVAL/"results/complete.json",dict(files={f:sha(EVAL/f) for f in files},new_outputs=420,anchor_checks=12,external_reference_outputs=840))

if __name__=="__main__":
    torch.set_num_threads(1)
    run("strict")
