"""Seal seed-A nominations before opening seed-B groups, then estimate replication."""
import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path

import numpy as np

SCORING = Path(__file__).resolve().parents[1]
STAGE = SCORING.parent
BOOT_SEED = 271127
DRAWS = 20000


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    part = path.with_suffix(path.suffix+".part")
    part.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    part.replace(path)


def seal(name,payload):
    file = SCORING/"locks"/(name+".json")
    digest = SCORING/"locks"/(name+".sha256")
    if digest.is_file():
        assert sha(file)==digest.read_text().strip()
        existing = read(file)
        assert {k:v for k,v in existing.items() if k != "sealed_utc"} == payload
    else:
        assert not list((STAGE/"scores/b").glob("*.json"))
        write(file,dict(**payload,sealed_utc=datetime.now(timezone.utc).isoformat()))
        digest.write_text(sha(file)+"\n")
    return read(file)


def locked(name):
    p = SCORING/"locks"/(name+".json")
    assert sha(p)==p.with_suffix(".sha256").read_text().strip()
    return read(p)


def by_seed(seed,rows):
    proof = read(STAGE/"sampling"/(seed+"_complete.json"))
    names = {row["image"] for row in rows}
    assert set(proof["files"]) == set(proof["eligibility"]) == names
    groups = {}
    for row in rows:
        name = row["image"]
        path = STAGE/"scores"/seed/(name+".json")
        g = read(path)
        assert g["rgb"]==row["rgb_sha256"] and g["alt_ids"]==row["alt_ids"]
        assert g["image"]==name and g["seed"]==seed
        assert g["source"]==proof["files"][name] and g["eligible"]==proof["eligibility"][name]
        if g["eligible"]:
            assert len(g["items"])==len(g["r2"])==len(g["r3"])==len(g["bonus"])==8
            assert len(g["r2_top"]) and len(g["r3_top"])
            assert all(len(i["other_caption_rr"])==len(row["alt_ids"]) for i in g["items"])
        groups[name]=g
    return groups


def nominate(rows):
    groups=by_seed("a",rows)
    eligible = [r for r in rows if groups[r["image"]]["eligible"] and r["alt_ids"]]
    positives = [r for r in eligible if groups[r["image"]]["opportunity_slots"]]
    order = sorted(positives,key=lambda r:r["rgb_sha256"])
    return seal("a_selection",dict(eligible=[r["image"] for r in eligible],selected=[r["image"] for r in order],
                   rgb={r["image"]:r["rgb_sha256"] for r in order},
                   groups={name:sha(STAGE/"scores/a"/(name+".json")) for name in groups},
                   source=sha(STAGE/"sampling/a_complete.json")))


def lock_training(rows):
    a = locked("a_selection")
    eligibility_file = STAGE/"sampling/b_eligibility.json"
    eligibility = read(eligibility_file)
    row_map = {r["image"]:r for r in rows}
    assert set(eligibility)==set(row_map) and all(type(v) is bool for v in eligibility.values())
    eligible_selected = [row_map[n] for n in a["selected"] if eligibility[n]]
    assert [r["rgb_sha256"] for r in eligible_selected] == sorted(r["rgb_sha256"] for r in eligible_selected)
    return seal("training",dict(a_selected=len(a["selected"]),both_eligible_selected=len(eligible_selected),
                   a_lock=sha(SCORING/"locks/a_selection.json"),b_eligibility_hash=sha(eligibility_file),
                   eligible_selected=[r["image"] for r in eligible_selected],rows=eligible_selected[:24]))


def joint_slots(detail,rr,top):
    # Independent diagnosis on D/E and F/G. Allow rounding error of the displayed
    # floating point averages in the ordinary-caption comparison only.
    return set(np.flatnonzero((detail>detail[top].mean())&(rr>=rr[top].mean()-1e-12)).tolist())


def group_metrics(g):
    items = g["items"]
    ab = (np.array([x["hard_margins"] for x in items])>0).mean(1)
    mrr = np.array([np.mean(x["other_caption_rr"]) for x in items])
    measures = dict(AB=ab,mrr=mrr)
    for k in ["DE","FG"]:
        if items[0][k] is not None:
            measures[k] = (np.array([x[k] for x in items])>0).mean(1)
    top = g["r2_top"]
    new = g["r3_top"]
    slots=set(g["opportunity_slots"])
    result=dict(opportunity=int(bool(slots)),achieved=int(g["achieved"]),
                r3_probability=len(slots&set(new))/len(new),
                r3_all_top_joint=int(set(new)<=slots),r3_winner_changed=int(set(top)!=set(new)))
    for key,z in measures.items():
        result[key+"_gain"] = float(z[new].mean()-z[top].mean())
        if key in ["DE","FG"]:
            independent = joint_slots(z,mrr,top)
            result[key+"_joint_candidate"] = int(bool(independent))
            result[key+"_joint_and_AB_same_slot"] = int(bool(independent&slots))
            result[key+"_joint_captured"] = int(bool(independent&set(new)))
    result["r3_mean_joint"]=int(result["AB_gain"]>0 and result["mrr_gain"]>=-1e-12)
    return result


def interval(draws):
    good=np.asarray(draws)[np.isfinite(draws)]
    return np.quantile(good,[.025,.975]).tolist() if len(good) else [None,None]


def ratio(values, weights):
    num,den=values.sum(0)
    draws=weights@values
    valid = draws[:,1]>0
    boot=np.full(len(draws),np.nan)
    np.divide(draws[:,0],draws[:,1],out=boot,where=valid)
    return dict(value=float(num/den) if den else None,interval=interval(boot),n=int(den),
                total=float(num),valid_draws=int(valid.sum())),boot


def report(rows,a,b,selected):
    paired = [r for r in rows if a[r["image"]]["eligible"] and b[r["image"]]["eligible"] and r["alt_ids"]]
    group_ids=sorted({(r["cohort"],r["pair"]) for r in paired})
    index={pair:i for i,pair in enumerate(group_ids)}
    outcomes = [(r,group_metrics(a[r["image"]]),group_metrics(b[r["image"]])) for r in paired]
    rng=np.random.default_rng(BOOT_SEED)
    weights=rng.multinomial(len(group_ids),np.full(len(group_ids),1/len(group_ids)),size=DRAWS)
    stats={}
    primary={}
    boots={}
    for arm in ["all","selected","unselected"]:
        subset=[(r,ma,mb) for r,ma,mb in outcomes if arm=="all" or (r["image"] in selected)==(arm=="selected")]
        stats[arm]={}
        for s,pos in [("a",1),("b",2)]:
            data={}
            for x in subset:
                i=index[(x[0]["cohort"],x[0]["pair"])]
                for k,v in x[pos].items():
                    if k not in data:
                        data[k]=np.zeros((len(index),2))
                    data[k][i]+=[v,1]
            metrics={}
            for k,arr in data.items():
                metrics[k],boot=ratio(arr,weights)
                if k=="opportunity" and s=="b":
                    primary[arm]=metrics[k]
                    boots[arm]=boot
            stats[arm][s]=metrics
    empty=dict(value=None,interval=[None,None],total=0,n=0,valid_draws=0)
    for arm in ("all","selected","unselected"):
        primary.setdefault(arm,empty)
    if all(k in boots for k in ["selected","unselected"]):
        delta = boots["selected"]-boots["unselected"]
        difference=dict(value=primary["selected"]["value"]-primary["unselected"]["value"],
                        interval=interval(delta),valid_draws=int(np.isfinite(delta).sum()))
    else:
        difference=dict(value=None,interval=[None,None],valid_draws=0)
    primary["difference"]=difference
    counts=dict(total=len(rows),pairs_with_both_eligible_image=len(group_ids),both_eligible_with_alt=len(paired),
                a_reached=sum(a[r["image"]]["eligible"] for r in rows),
                b_reached=sum(b[r["image"]]["eligible"] for r in rows),
                a_screened=sum(a[r["image"]]["eligible"] and bool(r["alt_ids"]) for r in rows),
                a_nominated=sum(r["image"] in selected for r in rows))
    return dict(primary=primary,stats=stats,counts=counts)


def finalize(rows):
    a_lock=locked("a_selection")
    train=locked("training")
    a,b=[by_seed(s,rows) for s in "ab"]
    selected=set(a_lock["selected"])
    assert selected=={name for name,g in a.items() if g["eligible"] and g["alt_ids"] and g["opportunity_slots"]}
    assert all(sha(STAGE/"scores/a"/(name+".json"))==val for name,val in a_lock["groups"].items())
    assert all(r["image"] in selected and b[r["image"]]["eligible"] for r in train["rows"])
    b_eligibility=read(STAGE/"sampling/b_eligibility.json")
    assert all(b[n]["eligible"]==v for n,v in b_eligibility.items())
    panel=report(rows,a,b,selected)
    low=panel["primary"]["difference"]["interval"][0]
    out=dict(primary=panel["primary"],replicated=bool(low is not None and low>0),training_count=len(train["rows"]),
             eligible_for_training=bool(low is not None and low>0 and len(train["rows"])>=2),
             bootstrap=dict(replicates=DRAWS,seed=BOOT_SEED,unit="(cohort,pair)",ratio="image weighted"),
             pooled=panel,cohorts={k:report([r for r in rows if r["cohort"]==k],a,b,selected)
                                  for k in sorted({r["cohort"] for r in rows})},
             locks={name:sha(SCORING/"locks"/(name+".json")) for name in ["a_selection","training"]},
             scorer_sha=sha(SCORING/"code/pool_reader.py"),
             inputs_sha=sha(SCORING/"inputs/manifest.json"))
    first_b=min((STAGE/"scores/b"/(name+".json")).stat().st_mtime for name in b)
    a_time=datetime.fromisoformat(a_lock["sealed_utc"]).timestamp()
    train_time=datetime.fromisoformat(train["sealed_utc"]).timestamp()
    assert a_time<=train_time<=first_b
    out["blinding"]=dict(a_sealed_utc=a_lock["sealed_utc"],training_sealed_utc=train["sealed_utc"],
                         first_b_score_utc=datetime.fromtimestamp(first_b,timezone.utc).isoformat(),
                         a_files_verified=len(a_lock["groups"]),b_score_count=len(b))
    write(SCORING/"results/summary.json",out)
    write(SCORING/"results/b_scores_sha256.json",{name:sha(STAGE/"scores/b"/(name+".json")) for name in b})
    archive=SCORING/"results/candidate_scores.jsonl.gz"
    n=0
    with archive.open("wb") as raw, gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as stream:
        for seed,groups in [("a",a),("b",b)]:
            for row in rows:
                g=groups[row["image"]]
                assert g["seed"]==seed
                stream.write((json.dumps(g,ensure_ascii=False,allow_nan=False)+"\n").encode())
                n+=1
    assert n==540
    write(SCORING/"results/score_archive.json",dict(file=archive.name,sha256=sha(archive),groups=n,
          images=len(rows),natural_candidate_groups=sum(g["eligible"] for groups in [a,b] for g in groups.values()),
          order="seed a, then seed b; frozen 270-image roster order within each"))
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["a","lock","b"])
    args=parser.parse_args()
    pool=read(SCORING/"inputs/rows.json")
    if args.phase=="a":
        print(len(nominate(pool)["selected"]))
    elif args.phase=="lock":
        print(len(lock_training(pool)["rows"]))
    else:
        print(finalize(pool)["primary"])
