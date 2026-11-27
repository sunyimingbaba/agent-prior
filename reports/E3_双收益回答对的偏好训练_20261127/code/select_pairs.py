"""Seal two actual sampled responses per image using visible training labels only."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


STAGE = Path(__file__).resolve().parents[1]
SOURCE = STAGE.parent/"E3_回填后双收益训练图筛选_20261127"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def ranked_pair(scores, visible, anchor=False):
    r2=scores["r2"]
    items=scores["items"]
    top=r2.index(max(r2))
    if not visible:
        return top, None
    count=[sum(x>0 for x in it["hard_margins"]) for it in items]
    weak=[min(it["hard_margins"]) for it in items]
    rr=[it["other_caption_rr"] for it in items]
    lower=[j for j in range(len(r2)) if r2[j]<r2[top]]
    if anchor:
        choices=[j for j in lower if count[j]<=count[top] and all(rr[j][k]<=rr[top][k] for k in visible)]
        order=lambda j:(r2[j],count[j],j)
    else:
        choices=[j for j in lower if count[j]>count[top] and all(rr[j][k]>=rr[top][k] for k in visible)]
        order=lambda j:(-count[j],-weak[j],-r2[j],j)
    return top,min(choices,key=order) if choices else None


def main():
    rows = sorted(read(SOURCE/"scoring/inputs/rows.json"),key=lambda r:r["rgb_sha256"])
    proof = read(SOURCE/"sampling/a_complete.json")
    score_sha=read(SOURCE/"scoring/locks/a_selection.json")["groups"]
    pools={"joint":[], "anchor":[]}
    gallery=STAGE.parent/"E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json"
    forbidden={r["rgb_sha256"] for r in read(gallery)}
    for row in rows:
        name=row["image"]
        assert row["rgb_sha256"] not in forbidden
        if len(row["alt_ids"])<2 or not proof["eligibility"][name]:
            continue
        rp=SOURCE/"sampling/raw/a"/(name+".pt")
        sp=SOURCE/"scores/a"/(name+".json")
        assert sha(sp)==score_sha[name]
        score=read(sp)
        assert sha(rp)==score["source"]==proof["files"][name]
        raw=torch.load(rp,weights_only=True,map_location="cpu")
        assert raw["eligible"] and raw["query"]==row["orig_query"] and raw["rgb"]==row["rgb_sha256"]
        assert raw["context"]["answer"]==row["feedback"] and raw["image"]==name
        for kind in pools:
            i,j=ranked_pair(score,range(len(row["alt_ids"])),anchor=kind=="anchor")
            if j is None:
                continue
            assert raw["paths"][i]!=raw["paths"][j],(name,i,j)
            record={k:row[k] for k in ("image","row","rgb_sha256","pair","cohort","orig_query","feedback","split")}
            record.update(kind=kind,source_seed="a",raw_sha256=score["source"],score_sha256=score_sha[name],
                          r2_winner=i,r2_loser=j,r3_winner=j if kind=="joint" else i,
                          r3_loser=i if kind=="joint" else j)
            pools[kind].append(record)
    joints=pools["joint"][:12]
    names={r["image"] for r in joints}
    anchors=[r for r in pools["anchor"] if r["image"] not in names][:len(joints)]
    n=min(len(joints),len(anchors))
    assert n>=2 and len({r["image"] for r in anchors[:n]+joints[:n]})==2*n
    selected=[r for pair in zip(anchors[:n],joints[:n]) for r in pair]
    lock=dict(rows=selected,eligible={k:len(v) for k,v in pools.items()},counts=dict(Counter(r["kind"] for r in selected)),
              source_raw_manifest=sha(SOURCE/"sampling/a_complete.json"),source_scores_manifest=sha(SOURCE/"scoring/locks/a_selection.json"),
              protocol=sha(STAGE/"实验口径.md"),selection_code=sha(Path(__file__)))
    path=STAGE/"inputs/train_pairs.json"
    body=json.dumps(lock,indent=2,ensure_ascii=False)+"\n"
    if path.exists():
        assert path.read_text()==body
    else:
        path.write_text(body)
    path.with_suffix(".sha256").write_text(sha(path)+"\n")
    print(lock["eligible"],lock["counts"],sha(path))


if __name__=="__main__":
    main()
