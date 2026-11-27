"""Check archived winners against the original per-candidate PyTorch CPU dot product."""
from compare import OLD, REPORTS, NATIVE, ROOT, LIBRARIES, read, load, graded_bonus, original_mapping
from pathlib import Path
import gzip
import json
import numpy as np
import torch
import hashlib


def main():
    orig=load(NATIVE/"features.pt")
    text=orig["features"].double()
    index={tuple(q):i for i,q in enumerate(orig["tokens"].tolist())}
    even=np.array([int(hashlib.sha256(",".join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in orig["tokens"].tolist()])
    match=original_mapping()
    datasets=[(OLD,LIBRARIES,ROOT/"candidates/逐组读数.jsonl.gz"),
              (REPORTS/"E3_R3首位配文保护与首句复验_20261126",{"history98":LIBRARIES["v122"].parent/"reward_confirm"},
               ROOT/"candidates/分差核对98逐组.jsonl.gz")]
    counts = {"groups":0, "original_score_equal":0,"bonus_equal":0,"r2_winners_equal":0,"r3_winners_equal":0}
    for source,libs,file in datasets:
        with gzip.open(file,"rt") as f:
            records={(r["image"],r["seed"],r["block"]):r for r in map(json.loads,f)}
        for tag,lib in libs.items():
            bank=load(lib/"records/text.pt")
            remap=[index[tuple(q)] for q in bank["native_tokens"].tolist()]
            for row in read(source/"inputs/real_roster.json"):
                if row.get("cohort",tag)!=tag:continue
                name=row["image"];hard_ids=np.array([bank["maps"][name][k] for k in "AB"]).ravel()
                ht=bank["hard"][hard_ids].double()
                pos=[remap[i] for i in bank["maps"][name]["native"]]
                mask=even.copy();mask[pos]=False
                for seed in "ab":
                    feat=load(source/"records/real"/seed/"features"/(name+".pt"))
                    native,h=[],[]
                    for v in feat["visual"][:16].double():
                        ss=text@v;hs=ht@v
                        native.append(float(ss[pos[0]]-ss[mask].max()))
                        h.append((hs[::2]-hs[1::2]).tolist())
                    for i in (0,1):
                        sl=slice(8*i,8*i+8);r=records[name,seed,i]
                        base=match(np.array(native[sl]),1/(1+np.array(feat["ranks"][sl])))
                        bonus=graded_bonus(base,np.array(h[sl]))
                        after=base+bonus
                        counts["groups"]+=1
                        counts["original_score_equal"]+=int(np.array_equal(r["r2"],base))
                        counts["bonus_equal"]+=int(np.array_equal(r["bonus"],bonus))
                        counts["r2_winners_equal"]+=int(np.flatnonzero(base==base.max()).tolist()==r["winners_r2"])
                        counts["r3_winners_equal"]+=int(np.flatnonzero(after==after.max()).tolist()==r["winners_r3"])
    (ROOT/"candidates/逐候选原运算复核.json").write_text(json.dumps(counts,indent=2)+"\n")
    assert len(set(counts.values()))==1,counts


if __name__=="__main__":
    torch.set_num_threads(1);main()
