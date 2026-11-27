"""Audit the visible second-turn texts where hidden-state retrieval favors a response."""
from collections import Counter
import json
import re
from pathlib import Path

import torch

from select_pairs import STAGE,SOURCE,read,ranked_pair

ANSWER = re.compile(r"<answer>\s*([^<]*?[^\s<][^<]*?)\s*</answer>",re.I)


def main():
    examples=[]
    counts=Counter()
    roster=read(SOURCE/"scoring/inputs/rows.json")
    selected={r["image"] for r in read(STAGE/"inputs/train_pairs.json")["rows"] if r["kind"]=="joint"}
    for r in roster:
        name=r["image"]
        if len(r["alt_ids"])<2:
            continue
        s=read(SOURCE/"scores/a"/(name+".json"))
        if not s["eligible"]:
            continue
        raw=torch.load(SOURCE/"sampling/raw/a"/(name+".pt"),map_location="cpu",weights_only=True)
        complete=[bool(ANSWER.search(t)) for t in raw["texts"]]
        counts.update(groups=1,candidates=8,well_formed=sum(complete),any_well_formed=int(any(complete)))
        i,j=ranked_pair(s,range(len(r["alt_ids"])))
        if j is None:
            continue
        items=s["items"]
        base=items[i]
        num=lambda a:sum(x>0 for x in a["hard_margins"])
        valid=[k for k,it in enumerate(items) if s["r2"][k]<s["r2"][i] and num(it)>num(base)
               and all(a>=b for a,b in zip(it["other_caption_rr"],base["other_caption_rr"]))]
        counts.update(joint_groups=1,joint_slots=len(valid),tagged_joint_slots=sum(complete[k] for k in valid),
                      any_tagged_challenge=int(any(complete[k] for k in valid)),selected_challenge_tagged=int(complete[j]))
        examples.append(dict(image=name,in_training=name in selected,attribute=r["attribute"],target=r["A"][0],
                             baseline=raw["texts"][i],chosen=raw["texts"][j],extra=raw["extra_texts"][j],
                             eligible_slots=[dict(slot=k,text=raw["texts"][k],extra=raw["extra_texts"][k]) for k in valid]))
    (STAGE/"validation/训练侧回答类型.json").write_text(json.dumps(dict(counts=dict(counts),examples=examples),
                                            ensure_ascii=False,indent=2)+"\n")
    print(dict(counts))


if __name__=="__main__":main()
