from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parents[1]
def path(p):return next(REPORTS.glob(p))
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")

def main():
    source=path("E3_普通配文*/inputs/real_roster.json")
    ref=path("E3_双收益回答*/validation98/fresh98/inputs/rows.json")
    native=path("E3-1_奖励一致性*/空间关系候选*/native/index.json")
    cols=read(native)["columns"]
    canonical={(r["attribute"],r["side"]):r for r in read(ref)}
    exclusions=[ref,path("E3_R3改选*/inputs/real_roster.json"),path("E3-2_UCM同起点*/inputs/train.json"),
                path("E3-2_UCM与RSITMD*/inputs/ucm.json"),path("E3_UCM多描述*/inputs/panel.json"),
                path("E3_回填后双收益*/scoring/locks/training.json"),path("E3_双收益回答*/inputs/train_pairs.json")]
    banned=set()
    for p in exclusions:
        data=read(p)
        banned.update(r["rgb_sha256"] for r in (data["rows"] if isinstance(data,dict) else data))
    by_pair=defaultdict(list)
    for r in read(source):by_pair[r["pair"]].append(r)
    groups=defaultdict(list)
    for pair,rows in by_pair.items():
        rows=sorted(rows,key=lambda r:r["side"])
        assert len(rows)==2 and [r["side"] for r in rows]==[0,1]
        if any(x["rgb_sha256"] in banned for x in rows):continue
        a=rows[0]["attribute"]
        if (a,0) not in canonical:continue
        if any(tuple(r["A"])!=tuple(canonical[a,r["side"]]["A"]) for r in rows):continue
        if any(rows[0][w]!=rows[1][w][::-1] for w in "ABDE"):continue
        if any(not ({cols[c] for c in r["captions"]}-{cols[r["orig_query"]]}) for r in rows):continue
        groups[a].append(rows)
    selected=[]
    for a,gs in sorted(groups.items()):
        for rs in sorted(gs,key=lambda v:tuple(sorted(r["rgb_sha256"] for r in v)))[:3]:
            selected.extend(dict(r,cohort="independent282") for r in rs)
    assert len(set(r["rgb_sha256"] for r in selected))==len(selected)
    roster=ROOT/"inputs/roster.json"
    dump(roster,selected)
    dump(ROOT/"inputs/roster_lock.json",dict(source=sha(source),roster=sha(roster),images=len(selected),attributes=len(groups),
         available_pairs={a:len(gs) for a,gs in groups.items()},excluded_sources={str(p.relative_to(REPORTS)):sha(p) for p in exclusions}))
    print(len(selected),len(groups),{a:len(gs) for a,gs in groups.items()})
    print("wording mismatches",[(r["attribute"],w) for r in selected for w in "ABDE" if r[w]!=canonical[r["attribute"],r["side"]][w]][:10])

if __name__=="__main__":main()
