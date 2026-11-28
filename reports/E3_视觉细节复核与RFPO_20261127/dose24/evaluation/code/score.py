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
POOL = STAGE.parents[1]/"E3_回填后双收益训练图筛选_20261127"
spec = spec_from_file_location("pool_score", POOL/"evaluation/code/score.py")
pool = module_from_spec(spec)
spec.loader.exec_module(pool)
pool.EVAL = EVAL
from score_final import RFPO, FORM, read, dump, sha, calculate, details, counts, alt_summary, summary_by_seed, paired

CONTROLS = ("r2",)

def get(seed,arm,gallery,inputs):
    name=f"{seed}_{arm}"
    source=STAGE/"models"/name
    folder=EVAL/"records"/name
    cp=read(source/"complete.json")
    roster=STAGE/"inputs/roster.json"
    rows=read(roster)["rows"]
    lock=STAGE/"inputs/train_lock.json"
    assert cp["steps"]==len(rows)==24 and cp["train"]==sha(roster) and cp["lock"]==sha(lock)
    assert cp["initial"]==read(lock)["starts"][seed] and cp["scores"]==sha(source/"steps.json")
    assert [r["image"] for r in read(source/"steps.json")]==[r["image"] for r in rows]
    assert {r["rgb_sha256"] for r in rows}.isdisjoint(r["rgb_sha256"] for r in gallery)
    visual,proof,hashes,meta=pool.inspect_records(folder,gallery,cp["final"],inputs)
    assert proof["decoder"]==sha(EVAL/"code/predict.py") and proof["protocol"]==sha(EVAL/"验收口径.md")
    return visual,meta,dict(archive=pool.pack(name,folder,hashes),adapter=cp["final"],
                           env={k:proof[k] for k in ("torch","device","cuda","source_hashes")})

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
    delta=result["first106"]["contrasts"]["strict-r2"]["DE"]
    mr=result["official"]["contrasts"]["strict-r2"]["mR"]
    result["formal_direction_check"]=dict(DE106=delta["mean"]>=0,mR210=mr["mean"]>=0)
    result["formal_directions_pass"]=all(result["formal_direction_check"].values())
    result["matched_actions_among_210"]=same
    dump(EVAL/"archives/index.json",assets)
    dump(EVAL/"results/逐图指标.json",raw)
    dump(EVAL/"results/单次检索对照.json",result)
    files=("archives/index.json","results/逐图指标.json","results/单次检索对照.json","code/score.py","code/predict.py","验收口径.md")
    dump(EVAL/"results/complete.json",dict(files={f:sha(EVAL/f) for f in files},new_outputs=840))

if __name__=="__main__":
    torch.set_num_threads(1)
    run("strict")
