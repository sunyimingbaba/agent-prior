"""Bind the frozen native caption IDs and exact, previously verified hard-text vectors."""
import json
from pathlib import Path

import torch

from select_panel import ROOT, REPORTS, path, read, sha, dump


def main():
    rows=read(ROOT/"inputs/roster.json")
    fresh=path("E3_双收益回答*/validation98/fresh98/inputs/rows.json").parent
    old=read(fresh/"manifest.json")
    for name,h in old["compiled"].items():assert sha(fresh/name)==h
    reference=read(fresh/"rows.json")
    features=torch.load(fresh/"vectors.pt",map_location="cpu",weights_only=True)
    canonical={}
    for r in reference:
        key=r["attribute"],r["side"]
        if key in canonical:
            q=canonical[key]
            assert all(q[m]==r[m] for m in "ABDE") and torch.equal(features[r["image"]],features[q["image"]])
        canonical[key]=r
    native=path("E3-1_奖励一致性*/空间关系候选*/native/index.json")
    cols=read(native)["columns"]
    bank={}
    for r in rows:
        ref=canonical[r["attribute"],r["side"]]
        assert all(ref[key]==r[key] for key in "ABDE")
        r["native_ids"]=[cols[t] for t in r["captions"]]
        r["alt_ids"]=sorted(set(r["native_ids"])-set([r["native_ids"][0]]))
        bank[r["image"]]=features[ref["image"]].clone()
    out=ROOT/"inputs"
    dump(out/"rows.json",rows)
    torch.save(bank,out/"vectors.pt")
    old.update(rows=len(rows),alt_images=sum(bool(r["alt_ids"]) for r in rows),fg_images=sum(len(v)==12 for v in bank.values()),
               compiled={n:sha(out/n) for n in ["rows.json","vectors.pt"]},sources={str(p.relative_to(REPORTS)):sha(p) for p in
                [out/"roster.json",native,fresh/"rows.json",fresh/"vectors.pt",fresh/"manifest.json"]})
    dump(out/"manifest.json",old)

if __name__=="__main__":main()
