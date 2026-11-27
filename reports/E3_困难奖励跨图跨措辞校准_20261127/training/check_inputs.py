"""Inspect original matched controls and run the proposed reward on archived training-image can​​didates."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
OLD = STAGE.parent / "E3_回填后双收益训练图筛选_20261127"
sys.path.insert(0, str(STAGE/"code"))
from reward import calibrated_scores, redistribute


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    roster = OLD/"scoring/locks/training.json"
    cases = read(roster)["rows"]
    assert len(cases)==24 and cases==sorted(cases,key=lambda r:r["rgb_sha256"])
    lock=read(OLD/"training/train_lock.json")
    result=dict(images=len(cases), roster_sha=digest(roster), train_lock=digest(OLD/"training/train_lock.json"),
                directions=digest(STAGE/"inputs/directions.npz"), control={}, archived_sampling={})
    with np.load(STAGE/"inputs/directions.npz") as directions:
        result["covered_images"]=sum(row["attribute"] in directions for row in cases)
        missing=[row["image"] for row in cases if row["attribute"] not in directions]
        result["uncovered_images"]=missing
        for seed in "ab":
            controls=[]
            for arm in ["r2","r3"]:
                parent=OLD/"models"/(seed+"_"+arm)
                proof=read(parent/"complete.json")
                assert proof["scores"]==digest(parent/"steps.json") and proof["train"]==digest(roster)
                assert proof["lock"]==result["train_lock"]
                assert proof["initial"]==lock["historical"][seed]["R0/adapter/adapter_model.safetensors"]
                steps=read(parent/"steps.json")
                assert [s["image"] for s in steps]==[c["image"] for c in cases]
                assert [s["seed"] for s in steps]==[(231000000 if seed=="a" else 241000000)+1000*j for j in range(len(cases))]
                result["control"][seed+"_"+arm]={"adapter":proof["final"],"steps":proof["steps"],"history":proof["scores"]}
                controls.append(steps[0])
            assert all(controls[0][k]==controls[1][k] for k in ["trajectories","continuations","r2","proxy_rank","native_margin","other_caption_rr"])
            rows=[]
            for c in cases:
                img=c["image"]
                hist=OLD/"sampling/raw"/seed/(img+".pt")
                scores=read(OLD/"scores"/seed/(img+".json"))
                assert digest(hist)==lock["contexts"][seed][img]==scores["source"]
                raw=torch.load(hist,weights_only=True,map_location="cpu")
                assert raw["eligible"] and raw["hidden"].shape==(8,3584)
                orig=np.asarray(scores["r2"])
                if c["attribute"] in directions:
                    evidence=calibrated_scores(raw["hidden"].double().numpy(),c["side"],directions[c["attribute"]])
                    adjusted=redistribute(orig,evidence,[x["other_caption_rr"] for x in scores["items"]])
                else:
                    adjusted=orig.copy()
                assert np.isfinite(adjusted).all() and np.isclose(orig.sum(),adjusted.sum())
                assert orig.min()<=adjusted.min()<=adjusted.max()<=orig.max()
                rows.append(dict(image=img,changed=bool((orig!=adjusted).any()),
                                 new_top=bool(not np.array_equal(adjusted==adjusted.max(),orig==orig.max()))))
            result["archived_sampling"][seed]=dict(changed=sum(r["changed"] for r in rows),new_top=sum(r["new_top"] for r in rows),rows=rows)
    out=STAGE/"training/固定训练输入核查.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(result["covered_images"],[(k,v["changed"],v["new_top"]) for k,v in result["archived_sampling"].items()])


if __name__=="__main__":
    main()
