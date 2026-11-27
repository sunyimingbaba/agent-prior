"""Predict image context side using frozen 98-image parameters and independent natural rollouts."""
from collections import defaultdict
from pathlib import Path
import json
import sys

import numpy as np

from check import fitted_axes
from independent import STAGE,REFERENCE,load,norm,pool,read,sha,stats

OUT=Path(__file__).resolve().parent
SOURCE=STAGE/"followup_pool"
FIXED=[151644,151644,198,18724,151645]


def main():
    axis,center=fitted_axes()
    source98=read(REFERENCE/"fresh98/inputs/rows.json")
    known={(r['attribute'],r['side'],tuple(r['A'])) for r in source98}
    rows=read(SOURCE/"inputs/rows.json")
    _,previous,_=pool()
    assert {r['rgb_sha256'] for r in rows}.isdisjoint(r['rgb_sha256'] for r in source98+previous)
    assert len(rows)==len({r['rgb_sha256'] for r in rows})==118
    assert all((r['attribute'],r['side'],tuple(r['A'])) in known for r in rows)
    manifest=read(SOURCE/"d_complete.json")
    assert len(manifest["files"])==len(rows)
    grouped=defaultdict(dict)
    original=defaultdict(dict)
    feedback=defaultdict(dict)
    records=[];fixed_single=eligible=0
    for r in rows:
        original[r["cohort"],r["pair"]][r["side"]]=r["orig_query"]
        feedback[r["cohort"],r["pair"]][r["side"]]=r["feedback"]
        name=r["image"];p=SOURCE/"raw/d"/(name+".pt")
        assert sha(p)==manifest["files"][name]
        z=load(p)
        assert z["rgb"]==r["rgb_sha256"] and z["query"]==r["orig_query"]
        if not z['eligible']:continue
        eligible+=1
        h=norm(z["hidden"].double().numpy())
        assert h.shape==(8,3584)
        attr=r["attribute"]
        estimate=float((h.mean(axis=0)-center[attr])@axis[attr])
        fixed=[j for j,seq in enumerate(z["paths"]) if seq==FIXED]
        constant=float((h[fixed[0]]-center[attr])@axis[attr]) if len(fixed) else None
        fixed_single+=int(bool(fixed))
        grouped[r["cohort"],r["pair"]][r["side"]]=(estimate,constant)
        records.append(dict(image=name,rgb=r["rgb_sha256"],side=r["side"],group=r["pair"],all_score=estimate,
                            fixed_score=constant,fixed_slot=(fixed[0] if fixed else None)))
    report={"possible_images":len(rows),"eligible_images":eligible,"fixed_any_images":fixed_single}
    hit=lambda v:1. if v>0 else 0. if v<0 else .5
    for j,label in enumerate(("all","matched_fixed")):
        panel=[]
        for pair,sides in grouped.items():
            if len(sides)!=2 or any(sides[i][j] is None for i in (0,1)):continue
            a,b=sides[0][j],sides[1][j]
            panel.append((pair,{"image_side_acc":(hit(a)+hit(-b))/2,"pair_order_acc":hit(a-b),
                                "raw_image_contrast":a-b,"both_classified":hit(a)*hit(-b)}))
        obj=stats(panel);obj["images"]=2*len(panel)
        obj["same_feedback_pairs"]=sum(feedback[pair][0]==feedback[pair][1] for pair,m in panel)
        for subset,share in (("identical_input_caption",True),("different_input_caption",False)):
            matches=[(pair,m) for pair,m in panel if (original[pair][0]==original[pair][1])==share]
            if matches:
                desc=stats(matches);desc['images']=2*len(matches)
                obj[subset]=desc
        report[label]=obj
    report['manifests']={"raw":sha(SOURCE/"d_complete.json"),"rows":sha(SOURCE/"inputs/rows.json"),"protocol":sha(OUT/"自然回答推广口径.md")}
    (OUT/"自然回答来源图侧.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"自然回答逐图预测.json").write_text(json.dumps(records,indent=2)+"\n")
    print(report)


if __name__=="__main__":main()
