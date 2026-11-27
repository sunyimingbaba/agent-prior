"""Audit prior image identities, mirror orientation and feature-label coverage before evaluation."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
REP=HERE.parents[1]
def read(q):return json.loads(q.read_bytes())
def path(q):return next(REP.glob(q))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def mirror(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["pair"]].append(r)
    attrs=defaultdict(lambda:defaultdict(set))
    for key,rs in groups.items():
        left,right=sorted(rs,key=lambda r:r["side"])
        assert left["side"]==0 and right["side"]==1 and left["attribute"]==right["attribute"]
        for k in "ABCDE":
            assert left[k]==list(reversed(right[k])),(key,k)
            attrs[left["attribute"]][k].add(tuple(left[k]))
    return attrs

def main():
    files=dict(
        controlled98=path("E3_R3首位配文*/inputs/real_roster.json"),
        natural270=path("E3_R3改选*/inputs/real_roster.json"),
        original24=path("E3-2_UCM同起点*/inputs/train.json"),
        screened24=path("E3_回填后双收益*/scoring/locks/training.json"),
        preference24=path("E3_双收益回答*/inputs/train_pairs.json"),
        formal210=path("E3-2_UCM与RSITMD*/inputs/ucm.json"),
        fine126=path("E3_UCM多描述*/inputs/panel.json"))
    data={}
    for k,v in files.items():
        rr=read(v)
        data[k]=rr["rows"] if isinstance(rr,dict) else rr
    pv={k:{q["rgb_sha256"] for q in rows} for k,rows in data.items()}
    kv={k:{q["pair"] for q in rows if "pair" in q} for k,rows in data.items()}
    full=mirror(data["natural270"]);old=mirror(data["controlled98"])
    intersection={}
    for k,ds in data.items():
        intersection[k]={j:dict(images=len(pv[k]&pv[j]),pair_ids=len(kv[k]&kv[j]) if kv[k] and kv[j] else None) for j in data}
    source=data["controlled98"];target=data["natural270"]
    attrs=sorted(set(r["attribute"] for r in source+target))
    table={}
    lookup={r["rgb_sha256"]:r for r in target}
    guess={tag:[lookup[r["rgb_sha256"]] for r in data[tag]] for tag in ["screened24","preference24"]}
    eligibility=read(path("E3_回填后双收益*/sampling/a_eligibility.json"))
    assert eligibility==read(path("E3_回填后双收益*/sampling/b_eligibility.json"))
    only2=read(path("E3_回填后双收益*/scoring/inputs/rows.json"))
    alt={r["image"]:len(r["alt_ids"]) for r in only2}
    for k in attrs:
        a=[r for r in source if r["attribute"]==k];t=[r for r in target if r["attribute"]==k]
        if a and t:
            assert all(old[k][l]==full[k][l] and len(old[k][l])==1 for l in "ABCDE")
        table[k]=dict(source98_pairs=len(a)//2,target270_pairs=len(t)//2,
                      both_natural_reached_images=sum(eligibility[r["image"]] for r in t),
                      both_reached_alt2=sum(eligibility[r["image"]] and alt[r["image"]]>=2 for r in t),
                      **{nm:sum(r["attribute"]==k for r in rr) for nm,rr in guess.items()})
    valid={r["rgb_sha256"] for r in target if r["attribute"] in old}
    used=pv["screened24"]|pv["preference24"];used_pairs=kv["screened24"]|kv["preference24"]
    coverage={x:sum(table[k][x] for k in old) for x in ["target270_pairs","both_natural_reached_images","both_reached_alt2","screened24","preference24"]}
    cover_fine=[]
    for r in data["fine126"]:
        if r["rgb_sha256"] in pv["formal210"] and r["attribute"] in old:
            norm=lambda k:tuple(r[k][::1 if r["side"]==0 else -1])
            if all(norm(w) in old[r["attribute"]][w] for w in "DE"):
                cover_fine.append(r)
    legacy_overlap={}
    for w in "AB":
        original={tuple(sorted(r[w])) for r in data["original24"]}
        oldwords={tuple(sorted(r[w])) for r in source}
        legacy_overlap[w]=len(original&oldwords)
    metrics=dict(roster_sha256={k:sha(p) for k,p in files.items()},overlaps=intersection,by_attribute=table,
          source_attribute_count=len(old),target_attribute_count=len(full),source_pair_frequency=dict(Counter(len([r for r in source if r["attribute"]==k])//2 for k in old)),
          supported=coverage,unsupported_attrs=sorted(set(full)-set(old)),old24_identical_wording=legacy_overlap,
          original_image_count=len(pv["original24"]),covered_fine106_exact_side_and_DE=len(cover_fine),
          screened_preference_unique_images=len(used),screened_preference_unique_pairs=len(used_pairs),
          target_outside_previous_two_training_pairs=sum(r["pair"] not in used_pairs for r in target),
          supported_outside_previous_two_training_pairs=sum(r["pair"] not in used_pairs and r["rgb_sha256"] in valid for r in target),
          viable_same_attribute_pair_heldout98=sum(len(v) for k,v in
                ((attr,[r for r in source if r["attribute"]==attr]) for attr in old) if len(v)>=4))
    (HERE/"历史图库与属性覆盖.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n")
    print({k:v for k,v in metrics.items() if k not in ["roster_sha256","overlaps","by_attribute"]})
if __name__=="__main__":main()
